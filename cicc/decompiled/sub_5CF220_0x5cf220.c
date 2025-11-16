// Function: sub_5CF220
// Address: 0x5cf220
//
__int64 __fastcall sub_5CF220(
        const __m128i *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        _DWORD *a8)
{
  int v10; // eax
  __m128i *v11; // rax
  __m128i *v12; // r12
  const __m128i *v13; // rbx
  int v14; // edx
  __int64 *m128i_i64; // r12
  __m128i *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdi
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int8 v24; // al
  bool v25; // cc
  __int64 v26; // rax
  __int64 v27; // r13
  __int64 v28; // rdi
  __m128i *v29; // rax
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 result; // rax
  __int64 v33; // [rsp-10h] [rbp-130h]
  const __m128i *v35; // [rsp+10h] [rbp-110h]
  int v38; // [rsp+34h] [rbp-ECh]
  __int64 v39; // [rsp+38h] [rbp-E8h]
  __int64 v40; // [rsp+38h] [rbp-E8h]
  __int64 v41; // [rsp+38h] [rbp-E8h]
  __m128i **v42; // [rsp+50h] [rbp-D0h]
  int v43; // [rsp+6Ch] [rbp-B4h] BYREF
  __int64 v44; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v45; // [rsp+78h] [rbp-A8h] BYREF
  __int64 v46; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v47; // [rsp+88h] [rbp-98h] BYREF
  _BYTE v48[144]; // [rsp+90h] [rbp-90h] BYREF

  v35 = a1;
  v44 = 0;
  v43 = 0;
  if ( !a1 )
    return 0;
  v10 = 0;
  v42 = (__m128i **)&v44;
  v38 = 0;
  do
  {
    if ( a2 && (v35->m128i_i8[11] & 1) == 0 )
      goto LABEL_6;
    if ( *(_BYTE *)(a3 + 80) == 19 )
    {
      v14 = 0;
      if ( v35->m128i_i8[8] == 80 )
        v14 = qword_4F077A8 > 0x9FC3u;
      if ( a7 != v14 )
        goto LABEL_6;
      if ( !v10 )
LABEL_14:
        sub_865900(a3);
    }
    else if ( !v10 )
    {
      goto LABEL_14;
    }
    v11 = (__m128i *)sub_727670();
    *v42 = v11;
    *v11 = _mm_loadu_si128(v35);
    v11[1] = _mm_loadu_si128(v35 + 1);
    v11[2] = _mm_loadu_si128(v35 + 2);
    v11[3] = _mm_loadu_si128(v35 + 3);
    v11[4] = _mm_loadu_si128(v35 + 4);
    (*v42)->m128i_i64[0] = 0;
    v12 = *v42;
    v13 = (const __m128i *)(*v42)[2].m128i_i64[0];
    if ( v13 )
    {
      m128i_i64 = v12[2].m128i_i64;
      sub_892150(v48);
      do
      {
        v16 = (__m128i *)sub_7276D0();
        *m128i_i64 = (__int64)v16;
        v17 = a4;
        *v16 = _mm_loadu_si128(v13);
        v16[1] = _mm_loadu_si128(v13 + 1);
        v16[2] = _mm_loadu_si128(v13 + 2);
        v18 = v13[1].m128i_i64[0];
        v19 = sub_869530(v18, a4, a5, (unsigned int)&v45, 0, (unsigned int)v48, (__int64)&v43);
        v23 = v33;
        if ( v19 )
        {
          v24 = v13->m128i_i8[10];
          v25 = (unsigned __int8)v24 <= 4u;
          if ( v24 != 4 )
          {
LABEL_21:
            if ( v25 )
            {
              if ( v24 == 3 )
              {
                v26 = v13[2].m128i_i64[1];
                if ( *(_BYTE *)(v26 + 173) == 12 )
                {
                  *(_QWORD *)(*m128i_i64 + 40) = v26;
                  sub_744F60(
                    *(_DWORD *)m128i_i64 + 40,
                    a6,
                    a4,
                    a5,
                    0,
                    (unsigned int)v48,
                    *m128i_i64 + 24,
                    (__int64)&v43);
                }
                else
                {
                  v27 = *(_QWORD *)(v26 + 144);
                  *(_QWORD *)(v26 + 144) = 0;
                  sub_7296C0(&v47);
                  v39 = *m128i_i64;
                  *(_QWORD *)(v39 + 40) = sub_740630(v13[2].m128i_i64[1]);
                  sub_729730((unsigned int)v47);
                  if ( v27 && (*(_BYTE *)(v27 - 8) & 1) != 0 )
                    *(_QWORD *)(v13[2].m128i_i64[1] + 144) = v27;
                }
              }
            }
            else
            {
              if ( v24 != 5 )
                sub_721090(v18);
              v30 = *m128i_i64;
              v40 = *(_QWORD *)(*m128i_i64 + 40);
              v46 = sub_724DC0(v40, v17, v20, v21, v22, v23);
              v31 = sub_7410C0(v40, a5, a4, 0, (int)v30 + 24, 4, (__int64)&v43, (__int64)v48, v46, (__int64)&v47);
              if ( !v31 )
              {
                if ( v47 )
                  v31 = sub_730690();
                else
                  v31 = sub_73A720(v46);
              }
              v41 = v31;
              sub_724E30(&v46);
              *(_QWORD *)(v30 + 40) = v41;
            }
            goto LABEL_27;
          }
          while ( 1 )
          {
            *(_QWORD *)(*m128i_i64 + 40) = v13[2].m128i_i64[1];
            sub_5C7EE0(*m128i_i64, a4, a5, a6, (__int64)v48, &v43);
LABEL_27:
            if ( v43 )
            {
              if ( !a8 && !v38 )
              {
                sub_6851C0(1867, &v13[1].m128i_u64[1]);
                v38 = 1;
              }
              (*v42)->m128i_i8[8] = 0;
            }
            v28 = v45;
            v17 = 0;
            *(_BYTE *)(*m128i_i64 + 11) &= ~1u;
            sub_867630(v28, 0);
            v18 = v45;
            if ( !(unsigned int)sub_866C00(v45) )
              break;
            m128i_i64 = (__int64 *)*m128i_i64;
            v29 = (__m128i *)sub_7276D0();
            *m128i_i64 = (__int64)v29;
            *v29 = _mm_loadu_si128(v13);
            v29[1] = _mm_loadu_si128(v13 + 1);
            v29[2] = _mm_loadu_si128(v13 + 2);
            v24 = v13->m128i_i8[10];
            v25 = (unsigned __int8)v24 <= 4u;
            if ( v24 != 4 )
              goto LABEL_21;
          }
        }
        else if ( (v13->m128i_i8[11] & 1) != 0 )
        {
          *(_BYTE *)(*m128i_i64 + 10) = 0;
        }
        v13 = (const __m128i *)v13->m128i_i64[0];
        m128i_i64 = (__int64 *)*m128i_i64;
      }
      while ( v13 );
      v12 = *v42;
    }
    v42 = (__m128i **)v12;
    v10 = 1;
LABEL_6:
    v35 = (const __m128i *)v35->m128i_i64[0];
  }
  while ( v35 );
  if ( v10 )
    sub_864110();
  result = v44;
  if ( v43 )
  {
    if ( a8 )
      *a8 = 1;
  }
  return result;
}
