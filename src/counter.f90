!
!   a Fortran routine to compute two point clustering
!   (c) Mehdi Rezaie
! module load python/3.6-anaconda
! f2py -m counter -c counter.f90
!
! Oct 22: change the algorithn due to a bug introduced by int
!         one must use floor
subroutine paircount(theta1, phi1, theta2, phi2, delta1, delta2, fpix1, fpix2, bins, auto, C, n1,n2, m)
integer,intent(in) :: n1, n2, m, auto
real(8),dimension(n1),intent(in)   :: theta1, phi1, delta1, fpix1
real(8),dimension(n2),intent(in)   :: theta2, phi2, delta2, fpix2
real(8),dimension(m+1),intent(in)  :: bins
real(8),dimension(2,m),intent(out) :: C


real(8) :: s, be
integer :: i,j,binid
real(8), dimension(n1) :: cost1,sint1,cosp1,sinp1
real(8), dimension(n2) :: cost2,sint2,cosp2,sinp2


!
cost1 = dcos(theta1)
sint1 = dsin(theta1)
cosp1 = dcos(phi1)
sinp1 = dsin(phi1)
!
cost2 = dcos(theta2)
sint2 = dsin(theta2)
cosp2 = dcos(phi2)
sinp2 = dsin(phi2)

C    = 0.0
s    = 0.0

if (auto .eq. 0) then
    do i = 1, n1
        do j = 1, n2
           s = sint1(i)*sint2(j)*(cosp1(i)*cosp2(j) + sinp1(i)*sinp2(j))+cost1(i)*cost2(j)
           be = bins(1) ! starts from the max sep angle
           binid = 0
           do while (s > be)
                binid = binid + 1
                be    = bins(binid+1)
           end do
           if ((binid .ge. 1) .and. (binid .le. m)) then
               C(1,binid) = C(1,binid) + delta1(i)*delta2(j)*fpix1(i)*fpix2(j)
               C(2,binid) = C(2,binid) + fpix1(i)*fpix2(j)
           end if
           s = 0
        end do
    end do
else if ((auto .eq. 1) .and. (n1 .eq. n2)) then
    do i = 1, n1
        do j = i+1, n1
           !print*, i, j
           s = sint1(i)*sint2(j)*(cosp1(i)*cosp2(j) + sinp1(i)*sinp2(j))+cost1(i)*cost2(j)
           be = bins(1) ! starts from the max sep angle
           binid = 0
           do while (s > be)
                binid = binid + 1
                be    = bins(binid+1)
           end do
           if ((binid .ge. 1) .and. (binid .le. m)) then
               C(1,binid) = C(1,binid) + delta1(i)*delta2(j)*fpix1(i)*fpix2(j)
               C(2,binid) = C(2,binid) + fpix1(i)*fpix2(j)
           end if
           s = 0
        end do
    end do
end if
return
end subroutine paircount
