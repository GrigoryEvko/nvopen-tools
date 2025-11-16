// Function: sub_14D1A80
// Address: 0x14d1a80
//
__int64 __fastcall sub_14D1A80(double (__fastcall *a1)(double, double), _QWORD *a2, double a3, double a4)
{
  int *v4; // rbx
  double v6; // [rsp+8h] [rbp-28h]

  feclearexcept(61);
  v4 = __errno_location();
  *v4 = 0;
  v6 = a1(a3, a4);
  if ( (unsigned int)(*v4 - 33) > 1 && !fetestexcept(29) )
    return sub_14D17B0(a2, (__int64)a2, v6);
  feclearexcept(61);
  *v4 = 0;
  return 0;
}
