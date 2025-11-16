// Function: sub_16977B0
// Address: 0x16977b0
//
void __fastcall sub_16977B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v7; // r12
  __int64 v8; // r13
  unsigned __int64 v9; // rax
  const __m128i *v10; // r11
  __m128i v11; // xmm1
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  __int8 *v14; // rax
  __m128i v15; // xmm0
  __m128i *v16; // rsi
  __m128i v17; // xmm2
  char *v18; // r12
  char *v19; // r13
  unsigned __int64 v20; // rax
  char *v21; // r11
  char *i; // rdi
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // rdx
  char *v25; // rcx
  char *v26; // rax
  __int64 v27; // r8
  __int64 v28; // rdx
  char *v29; // r12
  char *v30; // r13
  unsigned __int64 v31; // rax
  char *v32; // rax
  char *v33; // rdi
  __int64 v34; // rcx
  char *v35; // rdx
  char *v36; // rax
  __int64 v37; // rcx
  char *v38; // rdx
  char *v39; // rax
  __int64 v40; // rcx
  __int64 v41; // rsi
  char *v42; // rcx
  char *v43; // rax
  __int64 v44; // r8
  char *v45; // rdx
  char *v46; // r11
  char *j; // rdi
  unsigned __int64 v48; // rsi
  unsigned __int64 v49; // rdx
  char *v50; // rcx
  char *v51; // rax
  __int64 v52; // r8
  __int64 v53; // rdx
  __m128i v54; // [rsp-40h] [rbp-40h] BYREF

  if ( !*(_BYTE *)(a1 + 128) )
  {
    v7 = *(unsigned __int64 **)(a1 + 64);
    v8 = *(_QWORD *)(a1 + 56);
    if ( (unsigned __int64 *)v8 != v7 )
    {
      _BitScanReverse64(&v9, 0xAAAAAAAAAAAAAAABLL * (((__int64)v7 - v8) >> 3));
      sub_1697010(*(_QWORD *)(a1 + 56), *(__m128i **)(a1 + 64), 2LL * (int)(63 - (v9 ^ 0x3F)), a4, a5, a6);
      if ( (__int64)v7 - v8 <= 384 )
      {
        sub_1693D30(v8, v7);
      }
      else
      {
        sub_1693D30(v8, (unsigned __int64 *)(v8 + 384));
        for ( ; v7 != (unsigned __int64 *)v10; *(__m128i *)((char *)v16 + 8) = v17 )
        {
          v11 = _mm_loadu_si128(v10);
          v12 = v10->m128i_i64[0];
          v13 = v10[-2].m128i_u64[1];
          v54.m128i_i64[1] = v10[1].m128i_i64[0];
          v14 = &v10[-2].m128i_i8[8];
          v54.m128i_i64[0] = v11.m128i_i64[1];
          if ( v12 >= v13 )
          {
            v16 = (__m128i *)v10;
          }
          else
          {
            do
            {
              v15 = _mm_loadu_si128((const __m128i *)(v14 + 8));
              *((_QWORD *)v14 + 3) = v13;
              v16 = (__m128i *)v14;
              v14 -= 24;
              *(__m128i *)(v14 + 56) = v15;
              v13 = *(_QWORD *)v14;
            }
            while ( v12 < *(_QWORD *)v14 );
          }
          v17 = _mm_loadu_si128(&v54);
          v10 = (const __m128i *)((char *)v10 + 24);
          v16->m128i_i64[0] = v12;
        }
      }
    }
    v18 = *(char **)(a1 + 88);
    v19 = *(char **)(a1 + 80);
    if ( v19 != v18 )
    {
      _BitScanReverse64(&v20, (v18 - v19) >> 4);
      sub_1697370(*(char **)(a1 + 80), *(char **)(a1 + 88), 2LL * (int)(63 - (v20 ^ 0x3F)));
      if ( v18 - v19 <= 256 )
      {
        sub_1693B50(v19, v18);
      }
      else
      {
        sub_1693B50(v19, v19 + 256);
        for ( i = v21; i != v18; *((_QWORD *)v25 + 1) = v27 )
        {
          v23 = *(_QWORD *)i;
          v24 = *((_QWORD *)i - 2);
          v25 = i;
          v26 = i - 16;
          v27 = *((_QWORD *)i + 1);
          if ( *(_QWORD *)i < v24 )
          {
            do
            {
              *((_QWORD *)v26 + 2) = v24;
              v28 = *((_QWORD *)v26 + 1);
              v25 = v26;
              v26 -= 16;
              *((_QWORD *)v26 + 5) = v28;
              v24 = *(_QWORD *)v26;
            }
            while ( v23 < *(_QWORD *)v26 );
          }
          i += 16;
          *(_QWORD *)v25 = v23;
        }
      }
    }
    v29 = *(char **)(a1 + 112);
    v30 = *(char **)(a1 + 104);
    if ( v30 != v29 )
    {
      _BitScanReverse64(&v31, (v29 - v30) >> 4);
      sub_1697590(*(char **)(a1 + 104), *(char **)(a1 + 112), 2LL * (int)(63 - (v31 ^ 0x3F)));
      if ( v29 - v30 > 256 )
      {
        sub_1693AA0(v30, v30 + 256);
        for ( j = v46; v29 != j; *((_QWORD *)v50 + 1) = v52 )
        {
          v48 = *(_QWORD *)j;
          v49 = *((_QWORD *)j - 2);
          v50 = j;
          v51 = j - 16;
          v52 = *((_QWORD *)j + 1);
          if ( v49 > *(_QWORD *)j )
          {
            do
            {
              *((_QWORD *)v51 + 2) = v49;
              v53 = *((_QWORD *)v51 + 1);
              v50 = v51;
              v51 -= 16;
              *((_QWORD *)v51 + 5) = v53;
              v49 = *(_QWORD *)v51;
            }
            while ( v48 < *(_QWORD *)v51 );
          }
          j += 16;
          *(_QWORD *)v50 = v48;
        }
      }
      else
      {
        sub_1693AA0(v30, v29);
      }
      v32 = *(char **)(a1 + 104);
      v33 = *(char **)(a1 + 112);
      if ( v33 != v32 )
      {
        do
        {
          v32 += 16;
          if ( v33 == v32 )
            goto LABEL_37;
          v34 = *((_QWORD *)v32 - 2);
          v35 = v32 - 16;
        }
        while ( v34 != *(_QWORD *)v32 || *((_QWORD *)v32 - 1) != *((_QWORD *)v32 + 1) );
        if ( v33 != v35 )
        {
          v36 = v32 + 16;
          if ( v33 != v36 )
          {
            while ( 1 )
            {
              if ( *(_QWORD *)v36 == v34 && *((_QWORD *)v35 + 1) == *((_QWORD *)v36 + 1) )
              {
                v36 += 16;
                if ( v36 == v33 )
                  break;
              }
              else
              {
                *((_QWORD *)v35 + 2) = *(_QWORD *)v36;
                v37 = *((_QWORD *)v36 + 1);
                v36 += 16;
                v35 += 16;
                *((_QWORD *)v35 + 1) = v37;
                if ( v36 == v33 )
                  break;
              }
              v34 = *(_QWORD *)v35;
            }
          }
          v38 = v35 + 16;
          if ( v38 != v33 )
          {
            v39 = *(char **)(a1 + 112);
            v40 = v39 - v33;
            if ( v39 == v33 )
            {
              v45 = &v38[v40];
LABEL_36:
              *(_QWORD *)(a1 + 112) = v45;
              goto LABEL_37;
            }
            v41 = v40 >> 4;
            if ( v40 > 0 )
            {
              v42 = v38;
              v43 = v33;
              do
              {
                v44 = *(_QWORD *)v43;
                v42 += 16;
                v43 += 16;
                *((_QWORD *)v42 - 2) = v44;
                *((_QWORD *)v42 - 1) = *((_QWORD *)v43 - 1);
                --v41;
              }
              while ( v41 );
              v39 = *(char **)(a1 + 112);
              v40 = v39 - v33;
            }
            v45 = &v38[v40];
            if ( v45 != v39 )
              goto LABEL_36;
          }
        }
      }
    }
LABEL_37:
    *(_BYTE *)(a1 + 128) = 1;
  }
}
