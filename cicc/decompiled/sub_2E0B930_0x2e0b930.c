// Function: sub_2E0B930
// Address: 0x2e0b930
//
void __fastcall sub_2E0B930(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // rax
  __m128i *v7; // r13
  __m128i **v8; // r10
  _BYTE *v9; // rdi
  __int64 v10; // rsi
  unsigned __int64 v11; // r11
  unsigned __int64 v12; // r9
  __int64 v13; // rcx
  __m128i *v14; // rbx
  unsigned __int64 v15; // r14
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdx
  const __m128i *v18; // r8
  size_t v19; // rdx
  unsigned __int64 v20; // r15
  unsigned int v21; // esi
  __m128i *j; // rdx
  __int64 v23; // r14
  __int64 v24; // rax
  char *v25; // rcx
  char *v26; // rax
  char *v27; // rax
  const __m128i *v28; // r13
  unsigned __int64 v29; // rax
  char *v30; // rsi
  __m128i *v31; // rax
  signed __int64 v32; // rbx
  _BYTE *v33; // rax
  __m128i *v34; // rdx
  __int64 v35; // r15
  __m128i *v36; // r11
  unsigned __int64 v37; // rax
  const __m128i *v38; // rcx
  __m128i *i; // rdx
  const void *v40; // rsi
  _BYTE *v41; // rbx
  const void *v42; // rsi
  int v43; // [rsp-90h] [rbp-90h]
  const __m128i *v44; // [rsp-88h] [rbp-88h]
  __m128i **v45; // [rsp-80h] [rbp-80h]
  unsigned __int64 v46; // [rsp-80h] [rbp-80h]
  __m128i *v47; // [rsp-80h] [rbp-80h]
  __m128i **v48; // [rsp-78h] [rbp-78h]
  const __m128i *v49; // [rsp-78h] [rbp-78h]
  unsigned __int64 v50; // [rsp-78h] [rbp-78h]
  __m128i **v51; // [rsp-78h] [rbp-78h]
  const __m128i *v52; // [rsp-70h] [rbp-70h]
  __int64 v53; // [rsp-70h] [rbp-70h]
  __m128i **v54; // [rsp-70h] [rbp-70h]
  __m128i **v55; // [rsp-70h] [rbp-70h]
  unsigned __int64 v56; // [rsp-70h] [rbp-70h]
  __int8 *v57; // [rsp-68h] [rbp-68h]
  size_t v58; // [rsp-68h] [rbp-68h]
  __m128i **v59; // [rsp-68h] [rbp-68h]
  __m128i **v60; // [rsp-68h] [rbp-68h]
  char *v61; // [rsp-60h] [rbp-60h]
  unsigned __int64 *v62; // [rsp-60h] [rbp-60h]
  __m128i **v63; // [rsp-60h] [rbp-60h]
  _BYTE v64[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( (*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v6 = *(unsigned int *)(a1 + 40);
    *(_QWORD *)(a1 + 8) = 0;
    v7 = *(__m128i **)(a1 + 24);
    v8 = *(__m128i ***)a1;
    v9 = *(_BYTE **)(a1 + 16);
    if ( (_DWORD)v6 )
    {
      v10 = *((unsigned int *)v8 + 2);
      v11 = (unsigned __int64)*v8;
      v12 = 0xAAAAAAAAAAAAAAABLL * (((char *)v7 - v9) >> 3);
      v13 = 24 * v10;
      v14 = (__m128i *)((char *)*v8 + 24 * v10);
      if ( v12 >= v6 )
      {
        v24 = 3 * v6;
        v25 = &v9[8 * v24];
        if ( v14 != v7 )
        {
          v62 = (unsigned __int64 *)v8;
          v26 = (char *)memmove(&v9[8 * v24], v7, (char *)v14 - (char *)v7);
          v8 = (__m128i **)v62;
          v25 = v26;
          v11 = *v62;
        }
        *((_DWORD *)v8 + 2) = -1431655765 * ((__int64)&v25[(char *)v14 - (char *)v7 - v11] >> 3);
        v27 = *(char **)(a1 + 16);
      }
      else
      {
        v15 = v6 - v12;
        v61 = &v9[-v11];
        v16 = *((unsigned int *)v8 + 3);
        memset(v64, 0, 24);
        v17 = v6 - v12 + v10;
        if ( v14 == v7 )
        {
          v28 = (const __m128i *)v64;
          if ( v17 > v16 )
          {
            v42 = v8 + 2;
            if ( v11 > (unsigned __int64)v64 || v14 <= (__m128i *)v64 )
            {
              v60 = v8;
              sub_C8D5F0((__int64)v8, v42, v17, 0x18u, a5, v12);
              v8 = v60;
              v10 = *((unsigned int *)v60 + 2);
              v14 = (__m128i *)((char *)*v60 + 24 * v10);
            }
            else
            {
              v56 = v11;
              v59 = v8;
              sub_C8D5F0((__int64)v8, v42, v17, 0x18u, a5, v12);
              v8 = v59;
              v10 = *((unsigned int *)v59 + 2);
              v28 = (const __m128i *)&v64[(_QWORD)*v59 - v56];
              v14 = (__m128i *)((char *)*v59 + 24 * v10);
            }
          }
          if ( v15 )
          {
            v29 = v15;
            do
            {
              if ( v14 )
              {
                *v14 = _mm_loadu_si128(v28);
                v14[1].m128i_i64[0] = v28[1].m128i_i64[0];
              }
              v14 = (__m128i *)((char *)v14 + 24);
              --v29;
            }
            while ( v29 );
            LODWORD(v10) = *((_DWORD *)v8 + 2);
          }
          *((_DWORD *)v8 + 2) = v15 + v10;
        }
        else
        {
          v18 = (const __m128i *)v64;
          v57 = &v7->m128i_i8[-v11];
          if ( v17 > v16 )
          {
            v46 = v12;
            v40 = v8 + 2;
            v50 = v6;
            if ( v11 > (unsigned __int64)v64 || v14 <= (__m128i *)v64 )
            {
              v55 = v8;
              sub_C8D5F0((__int64)v8, v40, v17, 0x18u, (__int64)v64, v12);
              v8 = v55;
              v18 = (const __m128i *)v64;
              v12 = v46;
              v10 = *((unsigned int *)v55 + 2);
              v11 = (unsigned __int64)*v55;
              v7 = (__m128i *)&v57[(_QWORD)*v55];
              v6 = v50;
              v13 = 24 * v10;
              v14 = (__m128i *)((char *)*v55 + 24 * v10);
            }
            else
            {
              v54 = v8;
              v41 = &v64[-v11];
              sub_C8D5F0((__int64)v8, v40, v17, 0x18u, (__int64)v64, v12);
              v8 = v54;
              v12 = v46;
              v10 = *((unsigned int *)v54 + 2);
              v11 = (unsigned __int64)*v54;
              v18 = (const __m128i *)&v41[(_QWORD)*v54];
              v13 = 24 * v10;
              v7 = (__m128i *)&v57[(_QWORD)*v54];
              v6 = v50;
              v14 = (__m128i *)((char *)*v54 + 24 * v10);
            }
          }
          v19 = v13 - (_QWORD)v57;
          v20 = 0xAAAAAAAAAAAAAAABLL * ((v13 - (__int64)v57) >> 3);
          if ( v15 <= v20 )
          {
            v53 = 24 * v15;
            v34 = v14;
            v35 = 24 * (v10 + v12 - v6);
            v36 = (__m128i *)(v35 + v11);
            v37 = 0xAAAAAAAAAAAAAAABLL * ((v13 - v35) >> 3);
            if ( v37 + v10 > *((unsigned int *)v8 + 3) )
            {
              v43 = -1431655765 * ((v13 - v35) >> 3);
              v44 = v18;
              v47 = v36;
              v51 = v8;
              sub_C8D5F0((__int64)v8, v8 + 2, v37 + v10, 0x18u, (__int64)v18, v37 + v10);
              v8 = v51;
              LODWORD(v37) = v43;
              v18 = v44;
              v36 = v47;
              v10 = *((unsigned int *)v51 + 2);
              v34 = (__m128i *)((char *)*v51 + 24 * v10);
            }
            if ( v36 != v14 )
            {
              v38 = v36;
              do
              {
                if ( v34 )
                {
                  *v34 = _mm_loadu_si128(v38);
                  v34[1].m128i_i64[0] = v38[1].m128i_i64[0];
                }
                v38 = (const __m128i *)((char *)v38 + 24);
                v34 = (__m128i *)((char *)v34 + 24);
              }
              while ( v38 != v14 );
              LODWORD(v10) = *((_DWORD *)v8 + 2);
            }
            *((_DWORD *)v8 + 2) = v37 + v10;
            if ( v36 != v7 )
            {
              v45 = v8;
              v49 = v18;
              memmove((char *)v14 - (v35 - (_QWORD)v57), v7, v35 - (_QWORD)v57);
              v8 = v45;
              v18 = v49;
            }
            if ( v18 >= v7 && v18 < (__m128i *)((char *)*v8 + 24 * *((unsigned int *)v8 + 2)) )
              v18 = (const __m128i *)((char *)v18 + v53);
            if ( v15 )
            {
              for ( i = (__m128i *)((char *)v7 + v53); i != v7; v7[-1].m128i_i64[1] = v18[1].m128i_i64[0] )
              {
                v7 = (__m128i *)((char *)v7 + 24);
                *(__m128i *)((char *)v7 - 24) = _mm_loadu_si128(v18);
              }
            }
          }
          else
          {
            v21 = v15 + v10;
            *((_DWORD *)v8 + 2) = v21;
            if ( v7 != v14 )
            {
              v48 = v8;
              v52 = v18;
              v58 = v13 - (_QWORD)v57;
              memcpy((void *)(v11 + 24LL * v21 - v19), v7, v19);
              v8 = v48;
              v18 = v52;
              v19 = v58;
            }
            if ( v18 >= v7 && v18 < (__m128i *)((char *)*v8 + 24 * *((unsigned int *)v8 + 2)) )
              v18 = (const __m128i *)((char *)v18 + 24 * v15);
            if ( v20 )
            {
              for ( j = (__m128i *)((char *)v7 + v19); j != v7; v7[-1].m128i_i64[1] = v18[1].m128i_i64[0] )
              {
                v7 = (__m128i *)((char *)v7 + 24);
                *(__m128i *)((char *)v7 - 24) = _mm_loadu_si128(v18);
              }
            }
            v23 = v15 - v20;
            do
            {
              if ( v14 )
              {
                *v14 = _mm_loadu_si128(v18);
                v14[1].m128i_i64[0] = v18[1].m128i_i64[0];
              }
              v14 = (__m128i *)((char *)v14 + 24);
              --v23;
            }
            while ( v23 );
          }
        }
        v30 = &v61[**(_QWORD **)a1];
        *(_QWORD *)(a1 + 16) = v30;
        v27 = v30;
      }
      *(_QWORD *)(a1 + 24) = &v27[24 * *(unsigned int *)(a1 + 40)];
      sub_2E0B850((__m128i ***)a1);
    }
    else
    {
      v31 = *v8;
      v32 = (char *)*v8 + 24 * *((unsigned int *)v8 + 2) - (char *)v7;
      if ( (__m128i *)((char *)*v8 + 24 * *((unsigned int *)v8 + 2)) != v7 )
      {
        v63 = v8;
        v33 = memmove(v9, v7, (char *)*v8 + 24 * *((unsigned int *)v8 + 2) - (char *)v7);
        v8 = v63;
        v9 = v33;
        v31 = *v63;
      }
      *((_DWORD *)v8 + 2) = -1431655765 * ((&v9[v32] - (_BYTE *)v31) >> 3);
    }
  }
}
