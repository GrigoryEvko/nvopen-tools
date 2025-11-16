// Function: sub_DAC0E0
// Address: 0xdac0e0
//
__int64 __fastcall sub_DAC0E0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r12
  __int64 v8; // rdi
  unsigned int v9; // edx
  __int64 result; // rax
  _QWORD *v11; // [rsp+8h] [rbp-18h] BYREF

  v7 = a1 + 4;
  v8 = a1[8];
  v11 = v7;
  sub_DAB940(v8, (__int64)&v11, 1, a4, a5, a6);
  sub_C65A50(a1[8] + 1032LL, v7, v9);
  result = a1[3];
  if ( result )
  {
    if ( result != -4096 && result != -8192 )
      result = sub_BD60C0(a1 + 1);
    a1[3] = 0;
  }
  return result;
}
