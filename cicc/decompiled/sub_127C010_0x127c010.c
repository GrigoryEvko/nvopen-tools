// Function: sub_127C010
// Address: 0x127c010
//
void __fastcall sub_127C010(__int64 a1, _DWORD *a2)
{
  const char *v2; // r15
  __int64 v3; // rax
  __m128i si128; // xmm0
  size_t v5; // rdx
  __int64 v6; // rcx
  __m128i *v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rcx
  __m128i *v11; // rax
  __int64 v12; // rcx
  __m128i *v13; // rdx
  __int64 v14; // rax
  __m128i v15; // xmm0
  size_t v16; // rdx
  __int64 v17; // rcx
  __m128i *v18; // rax
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rcx
  __m128i *v22; // rax
  __m128i *v23; // rdx
  __int64 v24; // rcx
  __m128i *v25; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v26; // [rsp+28h] [rbp-A8h]
  __m128i v27; // [rsp+30h] [rbp-A0h] BYREF
  __m128i *v28; // [rsp+40h] [rbp-90h] BYREF
  __int64 v29; // [rsp+48h] [rbp-88h]
  __m128i v30; // [rsp+50h] [rbp-80h] BYREF
  __int64 v31; // [rsp+60h] [rbp-70h] BYREF
  __int64 v32; // [rsp+68h] [rbp-68h]
  __m128i v33; // [rsp+70h] [rbp-60h] BYREF
  __int64 v34; // [rsp+80h] [rbp-50h] BYREF
  __int64 v35; // [rsp+88h] [rbp-48h]
  _OWORD v36[4]; // [rsp+90h] [rbp-40h] BYREF

  v2 = *(const char **)(a1 + 8);
  if ( *(_BYTE *)(a1 + 172) == 2 || !v2 )
    return;
  if ( !HIDWORD(qword_4D046F0) || strcmp(v2, "clock") )
  {
    if ( !(_DWORD)qword_4D046F0 || strcmp(v2, "__ffs") && strcmp(v2, "__ffsll") )
      return;
    v31 = 18;
    v34 = (__int64)v36;
    v3 = sub_22409D0(&v34, &v31, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F101A0);
    v34 = v3;
    *(_QWORD *)&v36[0] = v31;
    *(_WORD *)(v3 + 16) = 10547;
    *(__m128i *)v3 = si128;
    v35 = v31;
    *(_BYTE *)(v34 + v31) = 0;
    v28 = &v30;
    v29 = 1;
    v30.m128i_i16[0] = 34;
    v5 = strlen(v2);
    if ( v5 <= 0x3FFFFFFFFFFFFFFELL )
    {
      v7 = (__m128i *)sub_2241490(&v28, v2, v5, v6);
      v31 = (__int64)&v33;
      if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
      {
        v33 = _mm_loadu_si128(v7 + 1);
      }
      else
      {
        v31 = v7->m128i_i64[0];
        v33.m128i_i64[0] = v7[1].m128i_i64[0];
      }
      v32 = v7->m128i_i64[1];
      v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
      v7->m128i_i64[1] = 0;
      v7[1].m128i_i8[0] = 0;
      v8 = 15;
      v9 = 15;
      if ( (__m128i *)v31 != &v33 )
        v9 = v33.m128i_i64[0];
      v10 = v32 + v35;
      if ( v32 + v35 <= v9 )
        goto LABEL_17;
      if ( (_OWORD *)v34 != v36 )
        v8 = *(_QWORD *)&v36[0];
      if ( v10 <= v8 )
      {
        v11 = (__m128i *)sub_2241130(&v34, 0, 0, v31, v32);
        v25 = &v27;
        v12 = v11->m128i_i64[0];
        v13 = v11 + 1;
        if ( (__m128i *)v11->m128i_i64[0] != &v11[1] )
          goto LABEL_18;
      }
      else
      {
LABEL_17:
        v11 = (__m128i *)sub_2241490(&v31, v34, v35, v10);
        v25 = &v27;
        v12 = v11->m128i_i64[0];
        v13 = v11 + 1;
        if ( (__m128i *)v11->m128i_i64[0] != &v11[1] )
        {
LABEL_18:
          v25 = (__m128i *)v12;
          v27.m128i_i64[0] = v11[1].m128i_i64[0];
LABEL_19:
          v26 = v11->m128i_i64[1];
          v11->m128i_i64[0] = (__int64)v13;
          v11->m128i_i64[1] = 0;
          v11[1].m128i_i8[0] = 0;
          if ( (__m128i *)v31 != &v33 )
            j_j___libc_free_0(v31, v33.m128i_i64[0] + 1);
          if ( v28 != &v30 )
            j_j___libc_free_0(v28, v30.m128i_i64[0] + 1);
          if ( (_OWORD *)v34 != v36 )
            j_j___libc_free_0(v34, *(_QWORD *)&v36[0] + 1LL);
          sub_6865F0(0xE9Du, a2, (__int64)v25, (__int64)"implement functionality with equivalent source code");
          if ( v25 != &v27 )
            j_j___libc_free_0(v25, v27.m128i_i64[0] + 1);
          return;
        }
      }
      v27 = _mm_loadu_si128(v11 + 1);
      goto LABEL_19;
    }
LABEL_55:
    sub_4262D8((__int64)"basic_string::append");
  }
  v34 = 18;
  v31 = (__int64)&v33;
  v14 = sub_22409D0(&v31, &v34, 0);
  v15 = _mm_load_si128((const __m128i *)&xmmword_3F10190);
  v31 = v14;
  v33.m128i_i64[0] = v34;
  *(_WORD *)(v14 + 16) = 10552;
  *(__m128i *)v14 = v15;
  v32 = v34;
  *(_BYTE *)(v31 + v34) = 0;
  v25 = &v27;
  v26 = 1;
  v27.m128i_i16[0] = 34;
  v16 = strlen(v2);
  if ( v16 > 0x3FFFFFFFFFFFFFFELL )
    goto LABEL_55;
  v18 = (__m128i *)sub_2241490(&v25, v2, v16, v17);
  v34 = (__int64)v36;
  if ( (__m128i *)v18->m128i_i64[0] == &v18[1] )
  {
    v36[0] = _mm_loadu_si128(v18 + 1);
  }
  else
  {
    v34 = v18->m128i_i64[0];
    *(_QWORD *)&v36[0] = v18[1].m128i_i64[0];
  }
  v35 = v18->m128i_i64[1];
  v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
  v18->m128i_i64[1] = 0;
  v18[1].m128i_i8[0] = 0;
  v19 = 15;
  v20 = 15;
  if ( (_OWORD *)v34 != v36 )
    v20 = *(_QWORD *)&v36[0];
  v21 = v35 + v32;
  if ( v35 + v32 > v20 )
  {
    if ( (__m128i *)v31 != &v33 )
      v19 = v33.m128i_i64[0];
    if ( v21 <= v19 )
    {
      v22 = (__m128i *)sub_2241130(&v31, 0, 0, v34, v35);
      v23 = v22 + 1;
      v28 = &v30;
      v24 = v22->m128i_i64[0];
      if ( (__m128i *)v22->m128i_i64[0] != &v22[1] )
        goto LABEL_38;
LABEL_49:
      v30 = _mm_loadu_si128(v22 + 1);
      goto LABEL_39;
    }
  }
  v22 = (__m128i *)sub_2241490(&v34, v31, v32, v21);
  v23 = v22 + 1;
  v28 = &v30;
  v24 = v22->m128i_i64[0];
  if ( (__m128i *)v22->m128i_i64[0] == &v22[1] )
    goto LABEL_49;
LABEL_38:
  v28 = (__m128i *)v24;
  v30.m128i_i64[0] = v22[1].m128i_i64[0];
LABEL_39:
  v29 = v22->m128i_i64[1];
  v22->m128i_i64[0] = (__int64)v23;
  v22->m128i_i64[1] = 0;
  v22[1].m128i_i8[0] = 0;
  if ( (_OWORD *)v34 != v36 )
    j_j___libc_free_0(v34, *(_QWORD *)&v36[0] + 1LL);
  if ( v25 != &v27 )
    j_j___libc_free_0(v25, v27.m128i_i64[0] + 1);
  if ( (__m128i *)v31 != &v33 )
    j_j___libc_free_0(v31, v33.m128i_i64[0] + 1);
  sub_6865F0(0xE9Du, a2, (__int64)v28, (__int64)"use clock64() instead");
  if ( v28 != &v30 )
    j_j___libc_free_0(v28, v30.m128i_i64[0] + 1);
}
