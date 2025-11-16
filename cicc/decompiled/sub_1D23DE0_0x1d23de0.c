// Function: sub_1D23DE0
// Address: 0x1d23de0
//
__int64 __fastcall sub_1D23DE0(
        _QWORD *a1,
        __int16 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  __int64 *v9; // r14
  __int64 *v10; // rbx
  __int64 v11; // rsi
  _QWORD *v12; // rsi
  __int64 v13; // rbx
  char v14; // r13
  int v15; // ecx
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  __int64 v19; // rax
  int v20; // [rsp+0h] [rbp-100h]
  int v25; // [rsp+28h] [rbp-D8h]
  __int64 *v26; // [rsp+38h] [rbp-C8h] BYREF
  unsigned __int64 v27[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v28[176]; // [rsp+50h] [rbp-B0h] BYREF

  if ( *(_BYTE *)(a4 + 16LL * (unsigned int)(a5 - 1)) == 111 )
  {
    v26 = 0;
    v14 = 0;
  }
  else
  {
    v27[0] = (unsigned __int64)v28;
    v27[1] = 0x2000000000LL;
    v26 = 0;
    sub_16BD430((__int64)v27, (unsigned __int16)~a2);
    sub_16BD4C0((__int64)v27, a4);
    v9 = a7;
    v10 = &a7[2 * a8];
    if ( a7 != v10 )
    {
      do
      {
        v11 = *v9;
        v9 += 2;
        sub_16BD4C0((__int64)v27, v11);
        sub_16BD430((__int64)v27, *((_DWORD *)v9 - 2));
      }
      while ( v10 != v9 );
    }
    v26 = 0;
    v12 = sub_1D17920((__int64)a1, (__int64)v27, a3, (__int64 *)&v26);
    if ( v12 )
    {
      v13 = sub_1D18310((__int64)a1, (__int64)v12, a3);
      if ( (_BYTE *)v27[0] != v28 )
        _libc_free(v27[0]);
      return v13;
    }
    if ( (_BYTE *)v27[0] != v28 )
      _libc_free(v27[0]);
    v14 = 1;
  }
  v13 = a1[26];
  v15 = *(_DWORD *)(a3 + 8);
  if ( v13 )
  {
    a1[26] = *(_QWORD *)v13;
  }
  else
  {
    v20 = *(_DWORD *)(a3 + 8);
    v19 = sub_145CBF0(a1 + 27, 112, 8);
    v15 = v20;
    v13 = v19;
  }
  v16 = *(_QWORD *)a3;
  v27[0] = v16;
  if ( v16 )
  {
    v25 = v15;
    sub_1623A60((__int64)v27, v16, 2);
    v15 = v25;
  }
  *(_QWORD *)v13 = 0;
  *(_QWORD *)(v13 + 8) = 0;
  *(_QWORD *)(v13 + 40) = a4;
  *(_QWORD *)(v13 + 16) = 0;
  *(_WORD *)(v13 + 24) = ~a2;
  *(_DWORD *)(v13 + 28) = -1;
  *(_QWORD *)(v13 + 32) = 0;
  *(_QWORD *)(v13 + 48) = 0;
  *(_DWORD *)(v13 + 56) = 0;
  *(_DWORD *)(v13 + 60) = a5;
  *(_DWORD *)(v13 + 64) = v15;
  v17 = (unsigned __int8 *)v27[0];
  *(_QWORD *)(v13 + 72) = v27[0];
  if ( v17 )
    sub_1623210((__int64)v27, v17, v13 + 72);
  *(_WORD *)(v13 + 80) &= 0xF000u;
  *(_WORD *)(v13 + 26) = 0;
  *(_QWORD *)(v13 + 88) = 0;
  *(_QWORD *)(v13 + 96) = 0;
  sub_1D23B60((__int64)a1, v13, (__int64)a7, a8);
  if ( v14 )
    sub_16BDA20(a1 + 40, (__int64 *)v13, v26);
  sub_1D172A0((__int64)a1, v13);
  return v13;
}
