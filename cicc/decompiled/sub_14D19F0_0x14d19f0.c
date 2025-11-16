// Function: sub_14D19F0
// Address: 0x14d19f0
//
__int64 __fastcall sub_14D19F0(double (__fastcall *a1)(double), _QWORD *a2, double a3)
{
  int *v3; // rbx
  double v5; // [rsp+8h] [rbp-28h]

  feclearexcept(61);
  v3 = __errno_location();
  *v3 = 0;
  v5 = a1(a3);
  if ( (unsigned int)(*v3 - 33) > 1 && !fetestexcept(29) )
    return sub_14D17B0(a2, (__int64)a2, v5);
  feclearexcept(61);
  *v3 = 0;
  return 0;
}
