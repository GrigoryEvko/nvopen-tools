// Function: sub_C9B170
// Address: 0xc9b170
//
__int64 __fastcall sub_C9B170(__int64 a1, __int64 a2, char *a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  __int64 result; // rax
  _QWORD *v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned __int64 v11; // r15
  __int64 v12; // rax
  int v13; // ecx
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rbx
  unsigned __int64 v21; // r15
  __int64 v22; // r14
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdx
  __m128i *v33; // r15
  __int64 v34; // rax
  __m128i v35; // xmm0
  _BYTE *v36; // rax
  _BYTE *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rbx
  __m128i v40; // xmm1
  size_t v41; // rdx
  _QWORD *v42; // [rsp+0h] [rbp-A0h]
  char *src; // [rsp+8h] [rbp-98h]
  _BYTE *srca; // [rsp+8h] [rbp-98h]
  __m128i *v45; // [rsp+10h] [rbp-90h]
  __int64 v46; // [rsp+18h] [rbp-88h]
  __m128i v47; // [rsp+20h] [rbp-80h] BYREF
  __m128i v48; // [rsp+30h] [rbp-70h] BYREF
  __m128i v49; // [rsp+40h] [rbp-60h] BYREF
  _OWORD v50[5]; // [rsp+50h] [rbp-50h] BYREF

  result = a2 - a1;
  src = a3;
  if ( a2 - a1 <= 768 )
    return result;
  if ( !a3 )
  {
    v22 = a2;
    goto LABEL_26;
  }
  v8 = (_QWORD *)a2;
  v42 = (_QWORD *)(a1 + 48);
  while ( 2 )
  {
    --src;
    v9 = *(v8 - 1);
    v10 = *(_QWORD *)(a1 + 88);
    v11 = a1
        + 16
        * (((0xAAAAAAAAAAAAAAABLL * (((__int64)v8 - a1) >> 4)
           + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v8 - a1) >> 4)) >> 63))
          & 0xFFFFFFFFFFFFFFFELL)
         + (__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)v8 - a1) >> 4)) / 2);
    v12 = *(_QWORD *)(v11 + 40);
    if ( v12 >= v10 )
    {
      if ( v10 > v9 )
        goto LABEL_23;
      if ( v12 > v9 )
        goto LABEL_7;
LABEL_22:
      sub_22415E0(
        a1,
        a1
      + 16
      * (((0xAAAAAAAAAAAAAAABLL * (((__int64)v8 - a1) >> 4) + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v8 - a1) >> 4)) >> 63))
        & 0xFFFFFFFFFFFFFFFELL)
       + (__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)v8 - a1) >> 4)) / 2));
      v28 = *(_QWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 32) = *(_QWORD *)(v11 + 32);
      v29 = *(_QWORD *)(v11 + 40);
      *(_QWORD *)(v11 + 32) = v28;
      v30 = *(_QWORD *)(a1 + 40);
      *(_QWORD *)(a1 + 40) = v29;
      *(_QWORD *)(v11 + 40) = v30;
      v18 = *(_QWORD *)(a1 + 88);
      v17 = *(v8 - 1);
      goto LABEL_8;
    }
    if ( v12 > v9 )
      goto LABEL_22;
    if ( v10 <= v9 )
    {
LABEL_23:
      sub_22415E0(a1, v42);
      v31 = *(_QWORD *)(a1 + 80);
      *(_QWORD *)(a1 + 80) = *(_QWORD *)(a1 + 32);
      v18 = *(_QWORD *)(a1 + 40);
      *(_QWORD *)(a1 + 32) = v31;
      v32 = *(_QWORD *)(a1 + 88);
      *(_QWORD *)(a1 + 88) = v18;
      *(_QWORD *)(a1 + 40) = v32;
      v17 = *(v8 - 1);
      goto LABEL_8;
    }
LABEL_7:
    sub_22415E0(a1, v8 - 6);
    v16 = *(_QWORD *)(a1 + 32);
    *(_QWORD *)(a1 + 32) = *(v8 - 2);
    *(v8 - 2) = v16;
    v17 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 40) = *(v8 - 1);
    *(v8 - 1) = v17;
    v18 = *(_QWORD *)(a1 + 88);
LABEL_8:
    v19 = *(_QWORD *)(a1 + 40);
    v20 = v42;
    v21 = (unsigned __int64)v8;
    while ( 1 )
    {
      v22 = (__int64)v20;
      if ( v19 < v18 )
        goto LABEL_15;
      v23 = v21 - 48;
      if ( v19 <= v17 )
      {
        v21 -= 48LL;
        if ( (unsigned __int64)v20 >= v23 )
          break;
        goto LABEL_14;
      }
      v24 = v21 - 96;
      do
      {
        v21 = v24;
        v24 -= 48LL;
      }
      while ( v19 > *(_QWORD *)(v24 + 88) );
      if ( (unsigned __int64)v20 >= v21 )
        break;
LABEL_14:
      sub_22415E0(v20, v21);
      v25 = v20[4];
      v20[4] = *(_QWORD *)(v21 + 32);
      v26 = *(_QWORD *)(v21 + 40);
      *(_QWORD *)(v21 + 32) = v25;
      v27 = v20[5];
      v20[5] = v26;
      v17 = *(_QWORD *)(v21 - 8);
      *(_QWORD *)(v21 + 40) = v27;
      v19 = *(_QWORD *)(a1 + 40);
LABEL_15:
      v18 = v20[11];
      v20 += 6;
    }
    sub_C9B170((_DWORD)v20, (_DWORD)v8, (_DWORD)src, v13, v14, v15, (char)v42);
    result = (__int64)v20 - a1;
    if ( (__int64)v20 - a1 > 768 )
    {
      if ( src )
      {
        v8 = v20;
        continue;
      }
LABEL_26:
      v33 = (__m128i *)(v22 - 32);
      v49.m128i_i8[0] = a7;
      sub_C9AFB0(a1, v22);
      srca = (_BYTE *)(a1 + 16);
      do
      {
        v45 = &v47;
        if ( (__m128i *)v33[-1].m128i_i64[0] == v33 )
        {
          v47 = _mm_loadu_si128(v33);
        }
        else
        {
          v45 = (__m128i *)v33[-1].m128i_i64[0];
          v47.m128i_i64[0] = v33->m128i_i64[0];
        }
        v34 = v33[-1].m128i_i64[1];
        v35 = _mm_loadu_si128(v33 + 1);
        v33[-1].m128i_i64[0] = (__int64)v33;
        v33[-1].m128i_i64[1] = 0;
        v46 = v34;
        v33->m128i_i8[0] = 0;
        v36 = *(_BYTE **)a1;
        v48 = v35;
        if ( v36 == srca )
        {
          v41 = *(_QWORD *)(a1 + 8);
          if ( v41 )
          {
            if ( v41 == 1 )
              v33->m128i_i8[0] = *(_BYTE *)(a1 + 16);
            else
              memcpy(v33, srca, v41);
            v41 = *(_QWORD *)(a1 + 8);
          }
          v33[-1].m128i_i64[1] = v41;
          v33->m128i_i8[v41] = 0;
          v37 = *(_BYTE **)a1;
        }
        else
        {
          v33[-1].m128i_i64[0] = (__int64)v36;
          v33[-1].m128i_i64[1] = *(_QWORD *)(a1 + 8);
          v33->m128i_i64[0] = *(_QWORD *)(a1 + 16);
          v37 = (_BYTE *)(a1 + 16);
          *(_QWORD *)a1 = srca;
        }
        *(_QWORD *)(a1 + 8) = 0;
        *v37 = 0;
        v38 = *(_QWORD *)(a1 + 32);
        v49.m128i_i64[0] = (__int64)v50;
        v33[1].m128i_i64[0] = v38;
        v33[1].m128i_i64[1] = *(_QWORD *)(a1 + 40);
        if ( v45 == &v47 )
        {
          v50[0] = _mm_load_si128(&v47);
        }
        else
        {
          v49.m128i_i64[0] = (__int64)v45;
          *(_QWORD *)&v50[0] = v47.m128i_i64[0];
        }
        v39 = (__int64)v33[-1].m128i_i64 - a1;
        v40 = _mm_load_si128(&v48);
        v47.m128i_i8[0] = 0;
        v49.m128i_i64[1] = v46;
        v50[1] = v40;
        result = sub_C96520(a1, 0, 0xAAAAAAAAAAAAAAABLL * (v39 >> 4), &v49);
        if ( (_OWORD *)v49.m128i_i64[0] != v50 )
          result = j_j___libc_free_0(v49.m128i_i64[0], *(_QWORD *)&v50[0] + 1LL);
        v33 -= 3;
      }
      while ( v39 > 48 );
    }
    return result;
  }
}
