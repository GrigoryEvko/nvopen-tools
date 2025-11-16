// Function: sub_1DB55F0
// Address: 0x1db55f0
//
void __fastcall sub_1DB55F0(__int64 a1)
{
  unsigned __int64 v2; // r15
  __m128i *v3; // r12
  _BYTE *v4; // r11
  __m128i **v5; // r9
  __int64 v6; // rsi
  __m128i *v7; // rdi
  unsigned __int64 v8; // r8
  __int64 v9; // rax
  __m128i *v10; // rbx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // r14
  char *v13; // r10
  size_t v14; // rdx
  unsigned __int64 v15; // rcx
  unsigned int v16; // esi
  __m128i *j; // rdx
  unsigned __int64 v18; // r14
  signed __int64 v19; // r11
  _BYTE *v20; // rcx
  _BYTE *v21; // rax
  __m128i *v22; // rax
  signed __int64 v23; // rbx
  _BYTE *v24; // rax
  __m128i *v25; // rdx
  __int64 v26; // r15
  const __m128i *v27; // r8
  unsigned __int64 v28; // rax
  const __m128i *v29; // rcx
  __m128i *i; // rax
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // [rsp-58h] [rbp-58h]
  char *v33; // [rsp-58h] [rbp-58h]
  unsigned __int64 v34; // [rsp-50h] [rbp-50h]
  int v35; // [rsp-50h] [rbp-50h]
  size_t v36; // [rsp-48h] [rbp-48h]
  __m128i **v37; // [rsp-48h] [rbp-48h]
  __m128i **v38; // [rsp-48h] [rbp-48h]
  __m128i **v39; // [rsp-48h] [rbp-48h]
  signed __int64 v40; // [rsp-40h] [rbp-40h]
  __m128i **v41; // [rsp-40h] [rbp-40h]
  __m128i **v42; // [rsp-40h] [rbp-40h]

  if ( (*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v2 = *(unsigned int *)(a1 + 40);
    *(_QWORD *)(a1 + 8) = 0;
    v3 = *(__m128i **)(a1 + 24);
    v4 = *(_BYTE **)(a1 + 16);
    v5 = *(__m128i ***)a1;
    if ( (_DWORD)v2 )
    {
      v6 = *((unsigned int *)v5 + 2);
      v7 = *v5;
      v8 = 0xAAAAAAAAAAAAAAABLL * (((char *)v3 - v4) >> 3);
      v9 = 24 * v6;
      v10 = (__m128i *)((char *)*v5 + 24 * v6);
      if ( v8 >= v2 )
      {
        v20 = &v4[24 * v2];
        if ( v10 != v3 )
        {
          v41 = v5;
          v21 = memmove(&v4[24 * v2], v3, (char *)v10 - (char *)v3);
          v5 = v41;
          v20 = v21;
          v7 = *v41;
        }
        *((_DWORD *)v5 + 2) = -1431655765 * ((&v20[(char *)v10 - (char *)v3] - (_BYTE *)v7) >> 3);
        v19 = *(_QWORD *)(a1 + 16);
      }
      else
      {
        v11 = *((unsigned int *)v5 + 3);
        v40 = v4 - (_BYTE *)v7;
        v12 = v2 - v8;
        v13 = (char *)((char *)v3 - (char *)v7);
        if ( v10 == v3 )
        {
          if ( v12 > v11 - v6 )
          {
            v38 = v5;
            sub_16CD150((__int64)v5, v5 + 2, v12 + v6, 24, v8, (int)v5);
            v5 = v38;
            v6 = *((unsigned int *)v38 + 2);
            v10 = (__m128i *)((char *)*v38 + 24 * v6);
          }
          if ( v12 )
          {
            v31 = v12;
            do
            {
              if ( v10 )
              {
                v10->m128i_i64[0] = 0;
                v10->m128i_i64[1] = 0;
                v10[1].m128i_i64[0] = 0;
              }
              v10 = (__m128i *)((char *)v10 + 24);
              --v31;
            }
            while ( v31 );
            LODWORD(v6) = *((_DWORD *)v5 + 2);
          }
          *((_DWORD *)v5 + 2) = v12 + v6;
        }
        else
        {
          if ( v12 + v6 > v11 )
          {
            v32 = 0xAAAAAAAAAAAAAAABLL * (((char *)v3 - v4) >> 3);
            v37 = v5;
            sub_16CD150((__int64)v5, v5 + 2, v12 + v6, 24, v8, (int)v5);
            v5 = v37;
            v13 = (char *)((char *)v3 - (char *)v7);
            v8 = v32;
            v6 = *((unsigned int *)v37 + 2);
            v7 = *v37;
            v3 = (__m128i *)&v13[(_QWORD)*v37];
            v9 = 24 * v6;
            v10 = (__m128i *)((char *)*v37 + 24 * v6);
          }
          v14 = v9 - (_QWORD)v13;
          v15 = 0xAAAAAAAAAAAAAAABLL * ((v9 - (__int64)v13) >> 3);
          if ( v12 <= v15 )
          {
            v25 = v10;
            v26 = 24 * (v6 + v8 - v2);
            v27 = (__m128i *)((char *)v7 + v26);
            v28 = 0xAAAAAAAAAAAAAAABLL * ((v9 - v26) >> 3);
            if ( v28 > (unsigned __int64)*((unsigned int *)v5 + 3) - v6 )
            {
              v33 = v13;
              v35 = v28;
              v39 = v5;
              sub_16CD150((__int64)v5, v5 + 2, v28 + v6, 24, (int)v27, (int)v5);
              v5 = v39;
              v27 = (__m128i *)((char *)v7 + v26);
              v13 = v33;
              LODWORD(v28) = v35;
              v6 = *((unsigned int *)v39 + 2);
              v25 = (__m128i *)((char *)*v39 + 24 * v6);
            }
            if ( v27 != v10 )
            {
              v29 = v27;
              do
              {
                if ( v25 )
                {
                  *v25 = _mm_loadu_si128(v29);
                  v25[1].m128i_i64[0] = v29[1].m128i_i64[0];
                }
                v29 = (const __m128i *)((char *)v29 + 24);
                v25 = (__m128i *)((char *)v25 + 24);
              }
              while ( v29 != v10 );
              LODWORD(v6) = *((_DWORD *)v5 + 2);
            }
            *((_DWORD *)v5 + 2) = v6 + v28;
            if ( v27 != v3 )
              memmove((char *)v10 - (v26 - (_QWORD)v13), v3, v26 - (_QWORD)v13);
            if ( v12 )
            {
              for ( i = (__m128i *)((char *)v3 + 24 * v12); i != v3; v3[-1].m128i_i64[1] = 0 )
              {
                v3->m128i_i64[0] = 0;
                v3 = (__m128i *)((char *)v3 + 24);
                v3[-1].m128i_i64[0] = 0;
              }
            }
          }
          else
          {
            v16 = v12 + v6;
            *((_DWORD *)v5 + 2) = v16;
            if ( v3 != v10 )
            {
              v34 = 0xAAAAAAAAAAAAAAABLL * ((v9 - (__int64)v13) >> 3);
              v36 = v9 - (_QWORD)v13;
              memcpy((char *)v7 + 24 * v16 - v14, v3, v14);
              v15 = v34;
              v14 = v36;
            }
            if ( v15 )
            {
              for ( j = (__m128i *)((char *)v3 + v14); j != v3; v3[-1].m128i_i64[1] = 0 )
              {
                v3->m128i_i64[0] = 0;
                v3 = (__m128i *)((char *)v3 + 24);
                v3[-1].m128i_i64[0] = 0;
              }
            }
            v18 = v12 - v15;
            do
            {
              if ( v10 )
              {
                v10->m128i_i64[0] = 0;
                v10->m128i_i64[1] = 0;
                v10[1].m128i_i64[0] = 0;
              }
              v10 = (__m128i *)((char *)v10 + 24);
              --v18;
            }
            while ( v18 );
          }
        }
        v19 = **(_QWORD **)a1 + v40;
        *(_QWORD *)(a1 + 16) = v19;
      }
      *(_QWORD *)(a1 + 24) = v19 + 24LL * *(unsigned int *)(a1 + 40);
      sub_1DB5510((__m128i ***)a1);
    }
    else
    {
      v22 = *v5;
      v23 = (char *)*v5 + 24 * *((unsigned int *)v5 + 2) - (char *)v3;
      if ( (__m128i *)((char *)*v5 + 24 * *((unsigned int *)v5 + 2)) != v3 )
      {
        v42 = *(__m128i ***)a1;
        v24 = memmove(*(void **)(a1 + 16), v3, (char *)*v5 + 24 * *((unsigned int *)v5 + 2) - (char *)v3);
        v5 = v42;
        v4 = v24;
        v22 = *v42;
      }
      *((_DWORD *)v5 + 2) = -1431655765 * ((&v4[v23] - (_BYTE *)v22) >> 3);
    }
  }
}
