// Function: sub_1D36D80
// Address: 0x1d36d80
//
__int64 *__fastcall sub_1D36D80(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        const void ***a4,
        int a5,
        double a6,
        double a7,
        __m128i a8,
        __int64 a9,
        __int128 a10)
{
  __int64 v11; // rsi
  int v12; // ecx
  __int64 v13; // r14
  unsigned __int8 *v14; // rsi
  __int64 *result; // rax
  __int64 *v16; // rbx
  __int64 *v17; // r14
  __int64 v18; // rsi
  __int64 v19; // rsi
  int v20; // ecx
  unsigned __int8 *v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rax
  __int16 v24; // [rsp+Ch] [rbp-F4h]
  int v27; // [rsp+28h] [rbp-D8h]
  __int64 *v28; // [rsp+28h] [rbp-D8h]
  int v29; // [rsp+28h] [rbp-D8h]
  int v30; // [rsp+28h] [rbp-D8h]
  int v31; // [rsp+28h] [rbp-D8h]
  __int64 *v32; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int8 *v33; // [rsp+38h] [rbp-C8h] BYREF
  unsigned __int64 v34[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v35[176]; // [rsp+50h] [rbp-B0h] BYREF

  v24 = a2;
  if ( a5 == 1 )
    return sub_1D359D0(a1, a2, a3, *(unsigned int *)a4, a4[1], 0, a6, a7, a8, a10);
  if ( LOBYTE(a4[2 * (unsigned int)(a5 - 1)]) == 111 )
  {
    v11 = *(_QWORD *)a3;
    v12 = *(_DWORD *)(a3 + 8);
    v34[0] = v11;
    if ( v11 )
    {
      v27 = v12;
      sub_1623A60((__int64)v34, v11, 2);
      v12 = v27;
    }
    v13 = a1[26];
    if ( v13 )
    {
      a1[26] = *(_QWORD *)v13;
    }
    else
    {
      v30 = v12;
      v22 = sub_145CBF0(a1 + 27, 112, 8);
      v12 = v30;
      v13 = v22;
    }
    *(_QWORD *)v13 = 0;
    *(_QWORD *)(v13 + 8) = 0;
    *(_WORD *)(v13 + 24) = v24;
    *(_QWORD *)(v13 + 16) = 0;
    *(_DWORD *)(v13 + 28) = -1;
    *(_QWORD *)(v13 + 32) = 0;
    *(_QWORD *)(v13 + 40) = a4;
    *(_QWORD *)(v13 + 48) = 0;
    *(_DWORD *)(v13 + 56) = 0;
    *(_DWORD *)(v13 + 60) = a5;
    *(_DWORD *)(v13 + 64) = v12;
    v14 = (unsigned __int8 *)v34[0];
    *(_QWORD *)(v13 + 72) = v34[0];
    if ( v14 )
      sub_1623210((__int64)v34, v14, v13 + 72);
    *(_WORD *)(v13 + 80) &= 0xF000u;
    *(_WORD *)(v13 + 26) = 0;
    sub_1D23B60((__int64)a1, v13, a10, *((__int64 *)&a10 + 1));
LABEL_10:
    sub_1D172A0((__int64)a1, v13);
    return (__int64 *)v13;
  }
  v34[0] = (unsigned __int64)v35;
  v34[1] = 0x2000000000LL;
  sub_16BD430((__int64)v34, (unsigned __int16)a2);
  sub_16BD4C0((__int64)v34, (__int64)a4);
  v16 = (__int64 *)a10;
  v17 = (__int64 *)(a10 + 16LL * *((_QWORD *)&a10 + 1));
  if ( (__int64 *)a10 != v17 )
  {
    do
    {
      v18 = *v16;
      v16 += 2;
      sub_16BD4C0((__int64)v34, v18);
      sub_16BD430((__int64)v34, *((_DWORD *)v16 - 2));
    }
    while ( v17 != v16 );
  }
  v32 = 0;
  result = sub_1D17920((__int64)a1, (__int64)v34, a3, (__int64 *)&v32);
  if ( !result )
  {
    v19 = *(_QWORD *)a3;
    v20 = *(_DWORD *)(a3 + 8);
    v33 = (unsigned __int8 *)v19;
    if ( v19 )
    {
      v29 = v20;
      sub_1623A60((__int64)&v33, v19, 2);
      v20 = v29;
    }
    v13 = a1[26];
    if ( v13 )
    {
      a1[26] = *(_QWORD *)v13;
    }
    else
    {
      v31 = v20;
      v23 = sub_145CBF0(a1 + 27, 112, 8);
      v20 = v31;
      v13 = v23;
    }
    *(_QWORD *)v13 = 0;
    v21 = v33;
    *(_QWORD *)(v13 + 8) = 0;
    *(_WORD *)(v13 + 24) = v24;
    *(_QWORD *)(v13 + 16) = 0;
    *(_DWORD *)(v13 + 28) = -1;
    *(_QWORD *)(v13 + 32) = 0;
    *(_QWORD *)(v13 + 40) = a4;
    *(_QWORD *)(v13 + 48) = 0;
    *(_DWORD *)(v13 + 56) = 0;
    *(_DWORD *)(v13 + 60) = a5;
    *(_DWORD *)(v13 + 64) = v20;
    *(_QWORD *)(v13 + 72) = v21;
    if ( v21 )
      sub_1623210((__int64)&v33, v21, v13 + 72);
    *(_WORD *)(v13 + 80) &= 0xF000u;
    *(_WORD *)(v13 + 26) = 0;
    sub_1D23B60((__int64)a1, v13, a10, *((__int64 *)&a10 + 1));
    sub_16BDA20(a1 + 40, (__int64 *)v13, v32);
    if ( (_BYTE *)v34[0] != v35 )
      _libc_free(v34[0]);
    goto LABEL_10;
  }
  if ( (_BYTE *)v34[0] != v35 )
  {
    v28 = result;
    _libc_free(v34[0]);
    return v28;
  }
  return result;
}
