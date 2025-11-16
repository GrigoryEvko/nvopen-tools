// Function: sub_FFDB80
// Address: 0xffdb80
//
void __fastcall sub_FFDB80(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  const __m128i *v7; // rbx
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __m128i *v12; // r15
  __m128i *v13; // rax
  __int64 v14; // r12
  __int64 v15; // rdi
  char v16; // dl
  __int64 v17; // r8
  __int64 v18; // rax
  __m128i v19; // xmm0
  unsigned __int64 v20; // rdx
  __m128i v21; // xmm0
  __m128i *v22; // rbx
  __m128i *v23; // rax
  _QWORD *v24; // rax
  __int64 *v25; // rdx
  __int64 *v26; // r13
  _BOOL4 v27; // r15d
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  __m128i v30; // xmm2
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // r8
  const __m128i *v34; // r15
  __m128i *v35; // rax
  __int64 v36; // r12
  _BYTE *v37; // r15
  __int64 v38; // r15
  const __m128i *v39; // [rsp+0h] [rbp-4A0h]
  const void *v40; // [rsp+8h] [rbp-498h]
  __m128i v41; // [rsp+30h] [rbp-470h] BYREF
  __int64 v42; // [rsp+40h] [rbp-460h]
  const __m128i *v43; // [rsp+48h] [rbp-458h]
  __m128i v44; // [rsp+50h] [rbp-450h] BYREF
  unsigned __int64 *v45; // [rsp+60h] [rbp-440h] BYREF
  __int64 v46; // [rsp+68h] [rbp-438h]
  _BYTE v47[128]; // [rsp+70h] [rbp-430h] BYREF
  __m128i *v48; // [rsp+F0h] [rbp-3B0h] BYREF
  __int64 v49; // [rsp+F8h] [rbp-3A8h]
  _BYTE v50[128]; // [rsp+100h] [rbp-3A0h] BYREF
  __int64 v51; // [rsp+180h] [rbp-320h] BYREF
  __int64 v52; // [rsp+188h] [rbp-318h] BYREF
  __int64 v53; // [rsp+190h] [rbp-310h]
  __int64 *v54; // [rsp+198h] [rbp-308h]
  __int64 *v55; // [rsp+1A0h] [rbp-300h]
  __int64 v56; // [rsp+1A8h] [rbp-2F8h]
  _BYTE v57[8]; // [rsp+1B0h] [rbp-2F0h] BYREF
  __m128i v58; // [rsp+1B8h] [rbp-2E8h]

  v6 = a1;
  v7 = (const __m128i *)a2;
  v8 = *(_QWORD *)(a1 + 544);
  if ( !v8 && !*(_QWORD *)(a1 + 552) )
    return;
  v9 = 2 * a3;
  LODWORD(v52) = 0;
  v48 = (__m128i *)v50;
  v49 = 0x800000000LL;
  v46 = 0x800000000LL;
  v53 = 0;
  v54 = &v52;
  v55 = &v52;
  v56 = 0;
  v45 = (unsigned __int64 *)v47;
  v43 = (const __m128i *)&a2[v9];
  if ( a2 != &a2[v9] )
  {
    v42 = a1;
    v40 = (const void *)(a1 + 16);
    while ( 1 )
    {
      v10 = v7->m128i_i64[0];
      v11 = v7->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL;
      v44.m128i_i64[0] = v10;
      v44.m128i_i64[1] = v11;
      if ( v10 != v11 )
      {
        if ( !v56 )
        {
          v12 = &v48[(unsigned int)v49];
          if ( v48 == v12 )
          {
            if ( (unsigned int)v49 <= 7uLL )
              goto LABEL_32;
          }
          else
          {
            v13 = v48;
            while ( v10 != v13->m128i_i64[0] || v11 != v13->m128i_i64[1] )
            {
              if ( v12 == ++v13 )
                goto LABEL_31;
            }
            if ( v12 != v13 )
              goto LABEL_12;
LABEL_31:
            if ( (unsigned int)v49 <= 7uLL )
            {
LABEL_32:
              v21 = _mm_load_si128(&v44);
              if ( (unsigned __int64)(unsigned int)v49 + 1 > HIDWORD(v49) )
              {
                v41 = v21;
                sub_C8D5F0((__int64)&v48, v50, (unsigned int)v49 + 1LL, 0x10u, (__int64)v48, a6);
                v21 = _mm_load_si128(&v41);
                v12 = &v48[(unsigned int)v49];
              }
              *v12 = v21;
              LODWORD(v49) = v49 + 1;
              goto LABEL_26;
            }
            v39 = v7;
            v22 = v48;
            v41.m128i_i64[0] = (__int64)v48[(unsigned int)v49].m128i_i64;
            do
            {
              v24 = sub_FFC140(&v51, &v52, (unsigned __int64 *)v22);
              v26 = v25;
              if ( v25 )
              {
                v27 = 1;
                if ( !v24 && v25 != &v52 )
                {
                  v28 = v25[4];
                  if ( v22->m128i_i64[0] >= v28 )
                  {
                    v27 = 0;
                    if ( v22->m128i_i64[0] == v28 )
                      v27 = v22->m128i_i64[1] < (unsigned __int64)v25[5];
                  }
                }
                v23 = (__m128i *)sub_22077B0(48);
                v23[2] = _mm_loadu_si128(v22);
                sub_220F040(v27, v23, v26, &v52);
                ++v56;
              }
              ++v22;
            }
            while ( (__m128i *)v41.m128i_i64[0] != v22 );
            v7 = v39;
          }
          LODWORD(v49) = 0;
          sub_FFC070((__int64)&v51, &v44);
LABEL_26:
          if ( (unsigned __int8)sub_FFB580(v42, v7->m128i_i64[0], v7->m128i_i64[1]) )
          {
            if ( *(_BYTE *)(v42 + 560) == 1 )
            {
              v29 = *(unsigned int *)(v42 + 8);
              v30 = _mm_loadu_si128(v7);
              v57[0] = 0;
              v31 = *(unsigned int *)(v42 + 12);
              v32 = *(_QWORD *)v42;
              v33 = v29 + 1;
              v34 = (const __m128i *)v57;
              v58 = v30;
              if ( v29 + 1 > v31 )
              {
                if ( v32 > (unsigned __int64)v57 || (unsigned __int64)v57 >= v32 + 32 * v29 )
                {
                  v38 = v42;
                  sub_C8D5F0(v42, v40, v33, 0x20u, v33, a6);
                  v32 = *(_QWORD *)v38;
                  v29 = *(unsigned int *)(v38 + 8);
                  v34 = (const __m128i *)v57;
                }
                else
                {
                  v36 = v42;
                  v37 = &v57[-v32];
                  sub_C8D5F0(v42, v40, v33, 0x20u, v33, a6);
                  v32 = *(_QWORD *)v36;
                  v29 = *(unsigned int *)(v36 + 8);
                  v34 = (const __m128i *)&v37[*(_QWORD *)v36];
                }
              }
              v35 = (__m128i *)(v32 + 32 * v29);
              *v35 = _mm_loadu_si128(v34);
              v35[1] = _mm_loadu_si128(v34 + 1);
              ++*(_DWORD *)(v42 + 8);
            }
            else
            {
              v18 = (unsigned int)v46;
              v19 = _mm_loadu_si128(v7);
              v20 = (unsigned int)v46 + 1LL;
              if ( v20 > HIDWORD(v46) )
              {
                v41 = v19;
                sub_C8D5F0((__int64)&v45, v47, v20, 0x10u, v17, a6);
                v18 = (unsigned int)v46;
                v19 = _mm_load_si128(&v41);
              }
              *(__m128i *)&v45[2 * v18] = v19;
              LODWORD(v46) = v46 + 1;
            }
          }
          goto LABEL_12;
        }
        sub_FFC070((__int64)&v51, &v44);
        if ( v16 )
          goto LABEL_26;
      }
LABEL_12:
      if ( v43 == ++v7 )
      {
        v6 = v42;
        a2 = v45;
        if ( *(_BYTE *)(v42 + 560) != 1 )
        {
          v8 = *(_QWORD *)(v42 + 544);
          goto LABEL_15;
        }
        goto LABEL_19;
      }
    }
  }
  v15 = 0;
  if ( *(_BYTE *)(v6 + 560) != 1 )
  {
    a2 = (unsigned __int64 *)v47;
LABEL_15:
    v43 = (const __m128i *)v8;
    if ( v8 )
    {
      sub_B26290((__int64)v57, a2, (unsigned int)v46, 1u);
      sub_B24D40((__int64)v43, (__int64)v57, 0);
      sub_B1A8B0((__int64)v57, (__int64)v57);
      a2 = v45;
    }
    v14 = *(_QWORD *)(v6 + 552);
    if ( v14 )
    {
      sub_B26B80((__int64)v57, a2, (unsigned int)v46, 1u);
      sub_B2A420(v14, (__int64)v57, 0);
      sub_B1AA80((__int64)v57, (__int64)v57);
      a2 = v45;
    }
LABEL_19:
    if ( a2 != (unsigned __int64 *)v47 )
      _libc_free(a2, a2);
    v15 = v53;
  }
  sub_FFB200(v15);
  if ( v48 != (__m128i *)v50 )
    _libc_free(v48, a2);
}
