// Function: sub_DAC150
// Address: 0xdac150
//
char __fastcall sub_DAC150(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r12
  __int64 v8; // rdi
  unsigned int v9; // edx
  __int64 v10; // rax
  _QWORD *v12; // [rsp+8h] [rbp-28h] BYREF

  v7 = a1 + 4;
  v8 = a1[8];
  v12 = v7;
  sub_DAB940(v8, (__int64)&v12, 1, a4, a5, a6);
  sub_C65A50(a1[8] + 1032LL, v7, v9);
  v10 = a1[3];
  if ( a2 != v10 )
  {
    if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
      sub_BD60C0(a1 + 1);
    a1[3] = a2;
    LOBYTE(v10) = a2 != 0;
    if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
      LOBYTE(v10) = sub_BD73F0((__int64)(a1 + 1));
  }
  return v10;
}
