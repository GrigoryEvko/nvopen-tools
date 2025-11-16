// Function: sub_833810
// Address: 0x833810
//
void __fastcall sub_833810(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i *a5, __int64 *a6)
{
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // r12
  __m128i *v10; // r15
  __int64 v11; // r13
  __int64 v12; // rbx
  __m128i *v13; // r14
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 *v18; // r9
  __int64 v19; // rax
  const __m128i *v20; // r14
  __m128i *v21; // r13
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 *v27; // r9
  __int64 v28; // [rsp+8h] [rbp-468h]
  __int64 v29; // [rsp+10h] [rbp-460h]
  __int64 *v30; // [rsp+10h] [rbp-460h]
  __int64 v31; // [rsp+18h] [rbp-458h]
  __m128i *v32; // [rsp+18h] [rbp-458h]
  __int64 v33; // [rsp+20h] [rbp-450h]
  __int64 *v34; // [rsp+20h] [rbp-450h]
  __int64 v35; // [rsp+20h] [rbp-450h]
  __int64 v36; // [rsp+28h] [rbp-448h]
  _BYTE v37[1000]; // [rsp+38h] [rbp-438h] BYREF
  __m128i *v38; // [rsp+420h] [rbp-50h]
  __int64 *v39; // [rsp+428h] [rbp-48h]
  __int64 v40; // [rsp+430h] [rbp-40h]

  v6 = qword_4F5F828;
  v39 = 0;
  v40 = 0;
  v7 = *(_QWORD *)(qword_4F5F828 + 3120);
  v38 = (__m128i *)v37;
  if ( v7 == *(_QWORD *)(qword_4F5F828 + 3112) )
  {
    v33 = qword_4F5F828;
    sub_8334C0(qword_4F5F828, a2, qword_4F5F828, a4, (__int64)a5, a6);
    v6 = v33;
  }
  v8 = *(_QWORD *)(v6 + 3104);
  v9 = v8 + 1032 * v7;
  if ( v9 )
  {
    *(_DWORD *)(v9 + 4) = 0;
    *(_QWORD *)(v9 + 1008) = 0;
    a5 = v38;
    *(_QWORD *)(v9 + 1016) = 0;
    v8 = v40;
    a6 = v39;
    *(_QWORD *)(v9 + 1024) = v40;
    if ( a5 == (__m128i *)v37 )
    {
      if ( (__int64)a6 <= 25 )
      {
        *(_DWORD *)(v9 + 4) = 1;
        a5 = (__m128i *)(v9 + 8);
      }
      else
      {
        v29 = v8;
        v31 = v6;
        v34 = a6;
        v19 = sub_823970(40LL * (_QWORD)a6);
        v8 = v29;
        v6 = v31;
        a6 = v34;
        a5 = (__m128i *)v19;
      }
      v20 = (const __m128i *)v37;
      v21 = a5;
      v22 = 0;
      if ( v8 > 0 )
      {
        do
        {
          if ( v21 )
          {
            *v21 = _mm_loadu_si128(v20);
            v21[1] = _mm_loadu_si128(v20 + 1);
            v21[2].m128i_i64[0] = v20[2].m128i_i64[0];
          }
          v23 = v20[2].m128i_i64[0];
          if ( v23 )
          {
            v28 = v22;
            v30 = a6;
            v32 = a5;
            v35 = v8;
            v36 = v6;
            sub_823A00(*(_QWORD *)v23, 16LL * (unsigned int)(*(_DWORD *)(v23 + 8) + 1), v6, v8, (__int64)a5, a6);
            sub_823A00(v23, 16, v24, v25, v26, v27);
            v22 = v28;
            a6 = v30;
            a5 = v32;
            v8 = v35;
            v6 = v36;
          }
          ++v22;
          v21 = (__m128i *)((char *)v21 + 40);
          v20 = (const __m128i *)((char *)v20 + 40);
        }
        while ( v8 != v22 );
      }
    }
    *(_QWORD *)(v9 + 1008) = a5;
    v10 = 0;
    *(_QWORD *)(v9 + 1016) = a6;
    v38 = 0;
    v39 = 0;
    v40 = 0;
    *(_QWORD *)(v6 + 3120) = v7 + 1;
LABEL_6:
    sub_823A00((__int64)v10, 40LL * (_QWORD)v39, v6, v8, (__int64)a5, a6);
    return;
  }
  v10 = v38;
  v11 = v40;
  *(_QWORD *)(v6 + 3120) = v7 + 1;
  v12 = 0;
  v13 = v10;
  if ( v11 > 0 )
  {
    do
    {
      v14 = v13[2].m128i_i64[0];
      if ( v14 )
      {
        sub_823A00(
          *(_QWORD *)v14,
          16LL * (unsigned int)(*(_DWORD *)(v14 + 8) + 1),
          *(unsigned int *)(v14 + 8),
          v8,
          (__int64)a5,
          a6);
        sub_823A00(v14, 16, v15, v16, v17, v18);
      }
      ++v12;
      v13 = (__m128i *)((char *)v13 + 40);
    }
    while ( v12 != v11 );
  }
  if ( v10 != (__m128i *)v37 )
    goto LABEL_6;
}
