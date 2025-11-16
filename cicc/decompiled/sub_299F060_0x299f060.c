// Function: sub_299F060
// Address: 0x299f060
//
void __fastcall sub_299F060(
        __m128i *a1,
        const __m128i *a2,
        __m128i *a3,
        __int64 a4,
        __int64 a5,
        __m128i *a6,
        __int64 a7,
        unsigned __int8 (__fastcall *a8)(__int64, __int64))
{
  __int64 v8; // rax
  const __m128i *v9; // r13
  __m128i *v10; // r12
  unsigned __int8 (__fastcall *v12)(__int64, __int64); // r15
  __int64 v13; // r14
  __int64 v14; // r9
  __int64 v15; // r15
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // r8
  const __m128i *v19; // r9
  __m128i *v20; // r10
  __int64 v21; // r12
  const __m128i *v22; // r11
  size_t v23; // r11
  __m128i *v24; // rax
  __int64 v25; // rax
  __m128i *v26; // r14
  __m128i v27; // xmm0
  __m128i v28; // xmm3
  __int64 v29; // rax
  size_t v30; // rcx
  unsigned int v31; // eax
  const __m128i *v32; // rax
  const __m128i *v33; // rax
  __m128i *v34; // [rsp+0h] [rbp-70h]
  const __m128i *v35; // [rsp+0h] [rbp-70h]
  __int64 v36; // [rsp+8h] [rbp-68h]
  size_t v37; // [rsp+8h] [rbp-68h]
  __int64 v38; // [rsp+8h] [rbp-68h]
  size_t v39; // [rsp+10h] [rbp-60h]
  __int64 v40; // [rsp+10h] [rbp-60h]
  __int64 v41; // [rsp+10h] [rbp-60h]
  __int64 v42; // [rsp+10h] [rbp-60h]
  size_t v43; // [rsp+10h] [rbp-60h]
  size_t v44; // [rsp+10h] [rbp-60h]
  int v45; // [rsp+10h] [rbp-60h]
  __int64 v46; // [rsp+10h] [rbp-60h]
  const __m128i *src; // [rsp+18h] [rbp-58h]
  const __m128i *srcb; // [rsp+18h] [rbp-58h]
  __m128i *srcc; // [rsp+18h] [rbp-58h]
  void *srcd; // [rsp+18h] [rbp-58h]
  __m128i *srca; // [rsp+18h] [rbp-58h]
  __m128i *srce; // [rsp+18h] [rbp-58h]
  void *srcf; // [rsp+18h] [rbp-58h]
  void *srcg; // [rsp+18h] [rbp-58h]
  int srch; // [rsp+18h] [rbp-58h]
  const __m128i *v56; // [rsp+28h] [rbp-48h]

  v8 = a5;
  v9 = a2;
  v10 = a1;
  v12 = a8;
  if ( a7 <= a5 )
    v8 = a7;
  if ( a4 <= v8 )
  {
LABEL_22:
    if ( v10 != v9 )
      memmove(a6, v10, (char *)v9 - (char *)v10);
    v26 = (__m128i *)((char *)a6 + (char *)v9 - (char *)v10);
    if ( a6 != v26 )
    {
      while ( a3 != v9 )
      {
        if ( v12((__int64)v9, (__int64)a6) )
        {
          v27 = _mm_loadu_si128(v9);
          v10 = (__m128i *)((char *)v10 + 56);
          v9 = (const __m128i *)((char *)v9 + 56);
          *(__m128i *)((char *)v10 - 56) = v27;
          *(__m128i *)((char *)v10 - 40) = _mm_loadu_si128((const __m128i *)((char *)v9 - 40));
          *(__m128i *)((char *)v10 - 24) = _mm_loadu_si128((const __m128i *)((char *)v9 - 24));
          v10[-1].m128i_i64[1] = v9[-1].m128i_i64[1];
          if ( v26 == a6 )
            return;
        }
        else
        {
          v28 = _mm_loadu_si128(a6);
          a6 = (__m128i *)((char *)a6 + 56);
          v10 = (__m128i *)((char *)v10 + 56);
          *(__m128i *)((char *)v10 - 56) = v28;
          *(__m128i *)((char *)v10 - 40) = _mm_loadu_si128((__m128i *)((char *)a6 - 40));
          *(__m128i *)((char *)v10 - 24) = _mm_loadu_si128((__m128i *)((char *)a6 - 24));
          v10[-1].m128i_i64[1] = a6[-1].m128i_i64[1];
          if ( v26 == a6 )
            return;
        }
      }
    }
    if ( v26 != a6 )
      memmove(v10, a6, (char *)v26 - (char *)a6);
  }
  else
  {
    v13 = a5;
    if ( a7 < a5 )
    {
      v14 = (__int64)a2;
      v15 = (__int64)a1;
      v16 = a4;
      while ( 1 )
      {
        src = (const __m128i *)v14;
        if ( v16 > v13 )
        {
          v21 = v16 / 2;
          v29 = sub_299EB50(v14, (__int64)a3, v15 + 56 * (v16 / 2), a8);
          v19 = src;
          v20 = (__m128i *)(v15 + 56 * (v16 / 2));
          v56 = (const __m128i *)v29;
          v18 = 0x6DB6DB6DB6DB6DB7LL * ((v29 - (__int64)src) >> 3);
        }
        else
        {
          v56 = (const __m128i *)(v14 + 56 * (v13 / 2));
          v17 = sub_299EBE0(v15, v14, (__int64)v56, a8);
          v18 = v13 / 2;
          v19 = src;
          v20 = (__m128i *)v17;
          v21 = 0x6DB6DB6DB6DB6DB7LL * ((v17 - v15) >> 3);
        }
        v16 -= v21;
        if ( v16 <= v18 || v18 > a7 )
        {
          if ( v16 > a7 )
          {
            v46 = v18;
            srch = (int)v20;
            v33 = sub_299DCF0(v20, v19, v56);
            v18 = v46;
            LODWORD(v20) = srch;
            v22 = v33;
          }
          else
          {
            v22 = v56;
            if ( v16 )
            {
              v30 = (char *)v19 - (char *)v20;
              if ( v20 != v19 )
              {
                v35 = v19;
                v38 = v18;
                v43 = (char *)v19 - (char *)v20;
                srce = v20;
                memmove(a6, v20, (char *)v19 - (char *)v20);
                v19 = v35;
                v18 = v38;
                v30 = v43;
                v20 = srce;
              }
              if ( v56 != v19 )
              {
                v44 = v30;
                srcf = (void *)v18;
                v31 = (unsigned int)memmove(v20, v19, (char *)v56 - (char *)v19);
                v30 = v44;
                v18 = (__int64)srcf;
                LODWORD(v20) = v31;
              }
              v22 = (const __m128i *)((char *)v56 - v30);
              if ( v30 )
              {
                v45 = (int)v20;
                srcg = (void *)v18;
                v32 = (const __m128i *)memmove((char *)v56 - v30, a6, v30);
                v18 = (__int64)srcg;
                LODWORD(v20) = v45;
                v22 = v32;
              }
            }
          }
        }
        else
        {
          v22 = v20;
          if ( v18 )
          {
            v23 = (char *)v56 - (char *)v19;
            if ( v56 != v19 )
            {
              v34 = v20;
              v36 = v18;
              v39 = (char *)v56 - (char *)v19;
              srcb = v19;
              memmove(a6, v19, (char *)v56 - (char *)v19);
              v20 = v34;
              v18 = v36;
              v23 = v39;
              v19 = srcb;
            }
            if ( v20 != v19 )
            {
              v37 = v23;
              v40 = v18;
              srcc = v20;
              memmove((char *)v56 - ((char *)v19 - (char *)v20), v20, (char *)v19 - (char *)v20);
              v23 = v37;
              v18 = v40;
              v20 = srcc;
            }
            if ( v23 )
            {
              v41 = v18;
              srcd = (void *)v23;
              v24 = (__m128i *)memmove(v20, a6, v23);
              v18 = v41;
              v23 = (size_t)srcd;
              v20 = v24;
            }
            v22 = (__m128i *)((char *)v20 + v23);
          }
        }
        v42 = v18;
        srca = (__m128i *)v22;
        sub_299F060(v15, (_DWORD)v20, (_DWORD)v22, v21, v18, (_DWORD)a6, a7, (__int64)a8);
        v25 = a7;
        v13 -= v42;
        if ( v13 <= a7 )
          v25 = v13;
        if ( v16 <= v25 )
        {
          v12 = a8;
          v10 = srca;
          v9 = v56;
          goto LABEL_22;
        }
        if ( v13 <= a7 )
          break;
        v14 = (__int64)v56;
        v15 = (__int64)srca;
      }
      v12 = a8;
      v10 = srca;
      v9 = v56;
    }
    if ( a3 != v9 )
      memmove(a6, v9, (char *)a3 - (char *)v9);
    sub_299EF60(
      v10,
      v9,
      a6,
      (__m128i *)((char *)a6 + (char *)a3 - (char *)v9),
      a3,
      (unsigned __int8 (__fastcall *)(const __m128i *, const __m128i *))v12);
  }
}
