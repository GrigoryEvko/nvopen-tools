// Function: sub_96A6F0
// Address: 0x96a6f0
//
__int64 __fastcall sub_96A6F0(double (__fastcall *a1)(double), __int64 a2, _QWORD *a3)
{
  int *v4; // rax
  int *v5; // rbx
  double v6; // xmm0_8
  double v8; // [rsp+8h] [rbp-28h]

  feclearexcept(61);
  v4 = __errno_location();
  *v4 = 0;
  v5 = v4;
  v6 = sub_C41B00(a2);
  v8 = a1(v6);
  if ( (unsigned int)(*v5 - 33) > 1 && !fetestexcept(29) )
    return sub_96A450(a3, v8);
  feclearexcept(61);
  *v5 = 0;
  return 0;
}
