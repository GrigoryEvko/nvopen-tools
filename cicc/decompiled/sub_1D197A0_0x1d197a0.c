// Function: sub_1D197A0
// Address: 0x1d197a0
//
_QWORD *__fastcall sub_1D197A0(
        __int64 a1,
        unsigned __int16 a2,
        __int64 a3,
        int a4,
        __int64 *a5,
        __int64 a6,
        __int16 a7)
{
  _QWORD *v7; // r15
  __int64 *v10; // r12
  __int64 v11; // rsi
  unsigned __int64 v12; // rdi
  __int64 v15; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v16; // [rsp+20h] [rbp-D0h] BYREF
  int v17; // [rsp+28h] [rbp-C8h]
  _QWORD v18[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v19[176]; // [rsp+40h] [rbp-B0h] BYREF

  v7 = 0;
  if ( *(_BYTE *)(a3 + 16LL * (unsigned int)(a4 - 1)) != 111 )
  {
    v18[1] = 0x2000000000LL;
    v18[0] = v19;
    sub_16BD430((__int64)v18, a2);
    sub_16BD4C0((__int64)v18, a3);
    v10 = &a5[2 * a6];
    while ( v10 != a5 )
    {
      v11 = *a5;
      a5 += 2;
      sub_16BD4C0((__int64)v18, v11);
      sub_16BD430((__int64)v18, *((_DWORD *)a5 - 2));
    }
    v15 = 0;
    v16 = 0;
    v17 = 0;
    v7 = sub_1D17920(a1, (__int64)v18, (__int64)&v16, &v15);
    if ( v16 )
      sub_161E7C0((__int64)&v16, v16);
    if ( !v7 )
    {
      v12 = v18[0];
      if ( (_BYTE *)v18[0] == v19 )
        return v7;
      goto LABEL_8;
    }
    sub_1D19330((__int64)v7, a7);
    v12 = v18[0];
    if ( (_BYTE *)v18[0] != v19 )
LABEL_8:
      _libc_free(v12);
  }
  return v7;
}
