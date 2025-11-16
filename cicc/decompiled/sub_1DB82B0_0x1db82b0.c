// Function: sub_1DB82B0
// Address: 0x1db82b0
//
void __fastcall sub_1DB82B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  _QWORD *v8; // r15
  __m128i v9; // xmm0
  __int64 v10; // r12
  __int64 *v11; // rcx
  __int64 *v12; // rdx
  __int64 *v13; // r14
  unsigned int v14; // esi
  unsigned int v15; // eax
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  _QWORD *v20; // rsi
  __int64 v21; // r9
  _QWORD *v22; // rax
  unsigned int v23; // ecx
  unsigned int v24; // esi
  __int64 v25; // rax
  __int64 *v26; // r12
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-88h]
  __int64 *v29; // [rsp+20h] [rbp-70h]
  __int64 v30; // [rsp+20h] [rbp-70h]
  __int64 *v31; // [rsp+28h] [rbp-68h]
  __int64 *v32; // [rsp+28h] [rbp-68h]
  __int64 v33; // [rsp+38h] [rbp-58h] BYREF
  __m128i v34; // [rsp+40h] [rbp-50h] BYREF
  __int64 v35; // [rsp+50h] [rbp-40h]

  v8 = *(_QWORD **)(a1 + 96);
  v33 = a1;
  v9 = _mm_loadu_si128((const __m128i *)&a7);
  v10 = a7;
  v35 = a8;
  v11 = v8 + 1;
  v34 = v9;
  v12 = (__int64 *)v8[2];
  if ( !v12 )
  {
    v13 = v8 + 1;
    goto LABEL_20;
  }
  v13 = v8 + 1;
  v14 = *(_DWORD *)((a7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)a7 >> 1) & 3;
  do
  {
    while ( 1 )
    {
      v15 = *(_DWORD *)((v12[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v12[4] >> 1) & 3;
      if ( v14 < v15
        || v14 <= v15
        && ((unsigned int)(v9.m128i_i64[1] >> 1) & 3 | *(_DWORD *)((v9.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)) < (*(_DWORD *)((v12[5] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v12[5] >> 1) & 3) )
      {
        break;
      }
      v12 = (__int64 *)v12[3];
      if ( !v12 )
        goto LABEL_8;
    }
    v13 = v12;
    v12 = (__int64 *)v12[2];
  }
  while ( v12 );
LABEL_8:
  if ( v11 == v13 )
  {
LABEL_20:
    v31 = (__int64 *)v8[3];
    if ( v31 == v13 )
    {
LABEL_14:
      v20 = sub_1DB7390(v8, v13, v34.m128i_i64);
      if ( v19 )
        sub_1DB3B70((__int64)v8, (__int64)v20, v19, &v34);
      return;
    }
    goto LABEL_11;
  }
  if ( v14 < (*(_DWORD *)((v13[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v13[4] >> 1) & 3) )
  {
    v31 = (__int64 *)v8[3];
    if ( v13 == v31 )
      goto LABEL_13;
  }
  else
  {
    v16 = sub_220EF30(v13);
    v11 = v8 + 1;
    v13 = (__int64 *)v16;
    v31 = (__int64 *)v8[3];
    if ( v31 == (__int64 *)v16 )
      goto LABEL_12;
  }
LABEL_11:
  v29 = v11;
  v17 = (_QWORD *)sub_220EFE0(v13);
  v11 = v29;
  if ( v35 == v17[6] )
  {
    v24 = *(_DWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v10 >> 1) & 3;
    if ( (*(_DWORD *)((v17[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v17[4] >> 1) & 3) <= v24
      && v24 <= (*(_DWORD *)((v17[5] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v17[5] >> 1) & 3) )
    {
      sub_1DB7AE0((__int64)&v33, (__int64)v17, v9.m128i_i64[1]);
      return;
    }
  }
LABEL_12:
  if ( v11 == v13 )
    goto LABEL_14;
LABEL_13:
  v18 = v13[6];
  if ( v35 != v18 )
    goto LABEL_14;
  v28 = (v9.m128i_i64[1] >> 1) & 3;
  if ( (*(_DWORD *)((v13[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v13[4] >> 1) & 3) > ((unsigned int)v28
                                                                                              | *(_DWORD *)((v9.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
    goto LABEL_14;
  v21 = (__int64)v13;
  do
  {
    if ( (__int64 *)v21 == v31 )
    {
      v13[4] = v10;
      sub_1DB7A40(*(__int64 **)(a1 + 96), v21, v13);
      goto LABEL_27;
    }
    v30 = v18;
    v22 = (_QWORD *)sub_220EFE0(v21);
    v18 = v30;
    v21 = (__int64)v22;
    v23 = (v10 >> 1) & 3 | *(_DWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  }
  while ( v23 <= (*(_DWORD *)((v22[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v22[4] >> 1) & 3) );
  if ( v23 > (*(_DWORD *)((v22[5] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v22[5] >> 1) & 3)
    || v30 != v22[6] )
  {
    v25 = sub_220EF30(v22);
    *(_QWORD *)(v25 + 32) = v10;
    v21 = v25;
  }
  *(_QWORD *)(v21 + 40) = v13[5];
  v32 = (__int64 *)v21;
  v26 = (__int64 *)sub_220EF30(v13);
  v27 = sub_220EF30(v32);
  sub_1DB7A40(*(__int64 **)(a1 + 96), v27, v26);
  v13 = v32;
LABEL_27:
  if ( (*(_DWORD *)((v9.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)v28) > (*(_DWORD *)((v13[5] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                         | (unsigned int)(v13[5] >> 1)
                                                                                         & 3) )
    sub_1DB7AE0((__int64)&v33, (__int64)v13, v9.m128i_i64[1]);
}
