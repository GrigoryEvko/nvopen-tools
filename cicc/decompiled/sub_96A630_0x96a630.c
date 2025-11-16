// Function: sub_96A630
// Address: 0x96a630
//
__int64 __fastcall sub_96A630(double (__fastcall *a1)(double, double), __int64 a2, __int64 a3, _QWORD *a4)
{
  int *v6; // rax
  int *v7; // rbx
  double v8; // xmm0_8
  double v10; // [rsp+8h] [rbp-38h]
  double v11; // [rsp+8h] [rbp-38h]

  feclearexcept(61);
  v6 = __errno_location();
  *v6 = 0;
  v7 = v6;
  v10 = sub_C41B00(a3);
  v8 = sub_C41B00(a2);
  v11 = a1(v8, v10);
  if ( (unsigned int)(*v7 - 33) > 1 && !fetestexcept(29) )
    return sub_96A450(a4, v11);
  feclearexcept(61);
  *v7 = 0;
  return 0;
}
