// Function: sub_200DF30
// Address: 0x200df30
//
__int64 __fastcall sub_200DF30(__int64 *a1, int a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v7; // rsi
  unsigned int v8; // r8d
  __int64 v9; // r9
  _BYTE *v10; // rax
  _BYTE *v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // r14
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // roff
  unsigned int v21; // [rsp+Ch] [rbp-104h]
  __int64 v22; // [rsp+10h] [rbp-100h]
  unsigned int v23; // [rsp+18h] [rbp-F8h]
  __int64 v24; // [rsp+20h] [rbp-F0h] BYREF
  int v25; // [rsp+28h] [rbp-E8h]
  _OWORD v26[2]; // [rsp+30h] [rbp-E0h] BYREF
  _BYTE *v27; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+58h] [rbp-B8h]
  _BYTE v29[176]; // [rsp+60h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(a3 + 72);
  v8 = *(_DWORD *)(a3 + 56);
  v24 = v7;
  if ( v7 )
  {
    v23 = v8;
    sub_1623A60((__int64)&v24, v7, 2);
    v8 = v23;
  }
  v25 = *(_DWORD *)(a3 + 64);
  switch ( v8 )
  {
    case 0u:
      sub_20BE530(
        (unsigned int)&v27,
        *a1,
        a1[1],
        a2,
        **(unsigned __int8 **)(a3 + 40),
        *(_QWORD *)(*(_QWORD *)(a3 + 40) + 8LL),
        0,
        0,
        a4,
        (__int64)&v24,
        0,
        1);
      v16 = (__int64)v27;
      goto LABEL_16;
    case 1u:
      v18 = *a1;
      v19 = 1;
      v26[0] = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a3 + 32));
      goto LABEL_21;
    case 2u:
      v18 = *a1;
      v19 = 2;
      v20 = *(_QWORD *)(a3 + 32);
      v26[0] = _mm_loadu_si128((const __m128i *)v20);
      v26[1] = _mm_loadu_si128((const __m128i *)(v20 + 40));
LABEL_21:
      sub_20BE530(
        (unsigned int)&v27,
        v18,
        a1[1],
        a2,
        **(unsigned __int8 **)(a3 + 40),
        *(_QWORD *)(*(_QWORD *)(a3 + 40) + 8LL),
        (__int64)v26,
        v19,
        a4,
        (__int64)&v24,
        0,
        1);
      v16 = (__int64)v27;
      goto LABEL_16;
  }
  v9 = v8;
  v28 = 0x800000000LL;
  v10 = v29;
  v27 = v29;
  if ( v8 > 8 )
  {
    v21 = v8;
    v22 = v8;
    sub_16CD150((__int64)&v27, v29, v8, 16, v8, v8);
    v10 = v27;
    v8 = v21;
    v9 = v22;
  }
  LODWORD(v28) = v8;
  v11 = &v10[16 * v9];
  do
  {
    if ( v10 )
    {
      *(_QWORD *)v10 = 0;
      *((_DWORD *)v10 + 2) = 0;
    }
    v10 += 16;
  }
  while ( v11 != v10 );
  v12 = 0;
  v13 = 0;
  do
  {
    v14 = *(_QWORD *)(a3 + 32);
    v15 = (__int64)v27;
    *(_QWORD *)&v27[v12] = *(_QWORD *)(v14 + v13);
    LODWORD(v14) = *(_DWORD *)(v14 + v13 + 8);
    v13 += 40;
    *(_DWORD *)(v15 + v12 + 8) = v14;
    v12 += 16;
  }
  while ( 40 * v9 != v13 );
  sub_20BE530(
    (unsigned int)v26,
    *a1,
    a1[1],
    a2,
    **(unsigned __int8 **)(a3 + 40),
    *(_QWORD *)(*(_QWORD *)(a3 + 40) + 8LL),
    (__int64)v27,
    (unsigned int)v28,
    a4,
    (__int64)&v24,
    0,
    1);
  v16 = *(_QWORD *)&v26[0];
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
LABEL_16:
  if ( v24 )
    sub_161E7C0((__int64)&v24, v24);
  return v16;
}
