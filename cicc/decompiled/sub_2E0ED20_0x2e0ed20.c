// Function: sub_2E0ED20
// Address: 0x2e0ed20
//
void __fastcall sub_2E0ED20(
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
  _QWORD *v10; // rcx
  _QWORD *v11; // rdx
  _QWORD *v12; // r14
  unsigned int v13; // esi
  unsigned int v14; // eax
  __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rdx
  _QWORD *v19; // rsi
  __int64 v20; // r9
  _QWORD *v21; // rax
  unsigned int v22; // ecx
  unsigned int v23; // esi
  __int64 v24; // rax
  _QWORD *v25; // r12
  __int64 v26; // rax
  __int64 v27; // [rsp+8h] [rbp-88h]
  _QWORD *v28; // [rsp+20h] [rbp-70h]
  __int64 v29; // [rsp+20h] [rbp-70h]
  _QWORD *v30; // [rsp+28h] [rbp-68h]
  __int64 v31; // [rsp+28h] [rbp-68h]
  __int64 v32; // [rsp+38h] [rbp-58h] BYREF
  __m128i v33; // [rsp+40h] [rbp-50h] BYREF
  __int64 v34; // [rsp+50h] [rbp-40h]

  v8 = *(_QWORD **)(a1 + 96);
  v32 = a1;
  v9 = _mm_loadu_si128((const __m128i *)&a7);
  v34 = a8;
  v10 = v8 + 1;
  v33 = v9;
  v11 = (_QWORD *)v8[2];
  if ( !v11 )
  {
    v12 = v8 + 1;
    goto LABEL_20;
  }
  v12 = v8 + 1;
  v13 = *(_DWORD *)((v9.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v9.m128i_i64[0] >> 1) & 3;
  do
  {
    while ( 1 )
    {
      v14 = *(_DWORD *)((v11[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v11[4] >> 1) & 3;
      if ( v14 > v13
        || v14 >= v13
        && ((unsigned int)(v9.m128i_i64[1] >> 1) & 3 | *(_DWORD *)((v9.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)) < (*(_DWORD *)((v11[5] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v11[5] >> 1) & 3) )
      {
        break;
      }
      v11 = (_QWORD *)v11[3];
      if ( !v11 )
        goto LABEL_8;
    }
    v12 = v11;
    v11 = (_QWORD *)v11[2];
  }
  while ( v11 );
LABEL_8:
  if ( v10 == v12 )
  {
LABEL_20:
    v30 = (_QWORD *)v8[3];
    if ( v30 == v12 )
    {
LABEL_14:
      v19 = sub_2E0D9D0(v8, v12, v33.m128i_i64);
      if ( v18 )
        sub_2E09B80((__int64)v8, (__int64)v19, v18, &v33);
      return;
    }
    goto LABEL_11;
  }
  if ( (*(_DWORD *)((v12[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v12[4] >> 1) & 3) > v13 )
  {
    v30 = (_QWORD *)v8[3];
    if ( v12 == v30 )
      goto LABEL_13;
  }
  else
  {
    v15 = sub_220EF30((__int64)v12);
    v10 = v8 + 1;
    v12 = (_QWORD *)v15;
    v30 = (_QWORD *)v8[3];
    if ( v30 == (_QWORD *)v15 )
      goto LABEL_12;
  }
LABEL_11:
  v28 = v10;
  v16 = (_QWORD *)sub_220EFE0((__int64)v12);
  v10 = v28;
  if ( v34 == v16[6] )
  {
    v23 = *(_DWORD *)((v9.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v9.m128i_i64[0] >> 1) & 3;
    if ( (*(_DWORD *)((v16[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v16[4] >> 1) & 3) <= v23
      && v23 <= (*(_DWORD *)((v16[5] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v16[5] >> 1) & 3) )
    {
      sub_2E0E620((__int64)&v32, (__int64)v16, v9.m128i_i64[1]);
      return;
    }
  }
LABEL_12:
  if ( v10 == v12 )
    goto LABEL_14;
LABEL_13:
  v17 = v12[6];
  if ( v34 != v17 )
    goto LABEL_14;
  v27 = (v9.m128i_i64[1] >> 1) & 3;
  if ( (*(_DWORD *)((v12[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v12[4] >> 1) & 3) > ((unsigned int)v27 | *(_DWORD *)((v9.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
    goto LABEL_14;
  v20 = (__int64)v12;
  do
  {
    if ( (_QWORD *)v20 == v30 )
    {
      v12[4] = v9.m128i_i64[0];
      sub_2E0E580(*(_QWORD **)(a1 + 96), v20, v12);
      goto LABEL_27;
    }
    v29 = v17;
    v21 = (_QWORD *)sub_220EFE0(v20);
    v17 = v29;
    v20 = (__int64)v21;
    v22 = (v9.m128i_i64[0] >> 1) & 3 | *(_DWORD *)((v9.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 24);
  }
  while ( v22 <= (*(_DWORD *)((v21[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v21[4] >> 1) & 3) );
  if ( v22 > (*(_DWORD *)((v21[5] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v21[5] >> 1) & 3)
    || v29 != v21[6] )
  {
    v24 = sub_220EF30((__int64)v21);
    *(_QWORD *)(v24 + 32) = v9.m128i_i64[0];
    v20 = v24;
  }
  *(_QWORD *)(v20 + 40) = v12[5];
  v31 = v20;
  v25 = (_QWORD *)sub_220EF30((__int64)v12);
  v26 = sub_220EF30(v31);
  sub_2E0E580(*(_QWORD **)(a1 + 96), v26, v25);
  v12 = (_QWORD *)v31;
LABEL_27:
  if ( (*(_DWORD *)((v9.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)v27) > (*(_DWORD *)((v12[5] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                         | (unsigned int)((__int64)v12[5] >> 1)
                                                                                         & 3) )
    sub_2E0E620((__int64)&v32, (__int64)v12, v9.m128i_i64[1]);
}
