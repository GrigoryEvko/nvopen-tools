// Function: sub_8334C0
// Address: 0x8334c0
//
__int64 __fastcall sub_8334C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rcx
  const __m128i *v9; // r12
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rbx
  __int64 i; // r15
  __int64 v15; // r13
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 *v19; // r9
  __int64 v21; // rax
  __m128i *v22; // r14
  __int64 v23; // r13
  __int64 v24; // rbx
  __int64 v25; // r15
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 *v29; // r9
  const __m128i *v30; // [rsp+8h] [rbp-88h]
  __int64 v31; // [rsp+10h] [rbp-80h]
  __int64 v32; // [rsp+18h] [rbp-78h]
  __int64 v33; // [rsp+20h] [rbp-70h]
  __int64 v34; // [rsp+28h] [rbp-68h]
  __int64 v35; // [rsp+30h] [rbp-60h]
  __int64 v37; // [rsp+40h] [rbp-50h]
  __int64 v38; // [rsp+48h] [rbp-48h]
  __int64 v39; // [rsp+48h] [rbp-48h]
  __int64 v40; // [rsp+48h] [rbp-48h]
  __int64 v41; // [rsp+50h] [rbp-40h]
  __int64 v42; // [rsp+58h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 3112);
  v32 = v6;
  v37 = *(_QWORD *)(a1 + 3120);
  v35 = *(_QWORD *)(a1 + 3104);
  if ( v6 > 1 )
  {
    v33 = v6 + (v6 >> 1) + 1;
    v7 = 1032 * v33;
LABEL_3:
    v34 = sub_823970(v7);
    goto LABEL_4;
  }
  v34 = a1 + 8;
  if ( *(_DWORD *)(a1 + 4) && a1 + 8 != v35 )
  {
    v33 = 2;
    v7 = 2064;
    goto LABEL_3;
  }
  v33 = 2;
  *(_DWORD *)(a1 + 4) = 1;
LABEL_4:
  v8 = v35;
  if ( v34 == v35 )
    goto LABEL_21;
  v41 = v34;
  v9 = (const __m128i *)(v35 + 8);
  v42 = 0;
  if ( v37 > 0 )
  {
    do
    {
      if ( v41 )
      {
        *(_DWORD *)(v41 + 4) = 0;
        *(_QWORD *)(v41 + 1008) = 0;
        *(_QWORD *)(v41 + 1016) = 0;
        v6 = v9[63].m128i_i64[1];
        *(_QWORD *)(v41 + 1024) = v6;
        v8 = v9[62].m128i_i64[1];
        v10 = v9[63].m128i_i64[0];
        if ( (const __m128i *)v8 == v9 )
        {
          if ( v10 <= 25 )
          {
            *(_DWORD *)(v41 + 4) = 1;
            v8 = v41 + 8;
          }
          else
          {
            v39 = v6;
            v21 = sub_823970(40 * v10);
            v6 = v39;
            v8 = v21;
          }
          v22 = (__m128i *)v8;
          if ( v6 > 0 )
          {
            v40 = v8;
            v23 = v6;
            v24 = 0;
            v31 = v10;
            v30 = v9;
            do
            {
              if ( v22 )
              {
                *v22 = _mm_loadu_si128(v9);
                v22[1] = _mm_loadu_si128(v9 + 1);
                v22[2].m128i_i64[0] = v9[2].m128i_i64[0];
              }
              v25 = v9[2].m128i_i64[0];
              if ( v25 )
              {
                sub_823A00(
                  *(_QWORD *)v25,
                  16LL * (unsigned int)(*(_DWORD *)(v25 + 8) + 1),
                  v6,
                  *(unsigned int *)(v25 + 8),
                  a5,
                  a6);
                sub_823A00(v25, 16, v26, v27, v28, v29);
              }
              ++v24;
              v22 = (__m128i *)((char *)v22 + 40);
              v9 = (const __m128i *)((char *)v9 + 40);
            }
            while ( v23 != v24 );
            v8 = v40;
            v10 = v31;
            v9 = v30;
          }
        }
        v38 = 0;
        *(_QWORD *)(v41 + 1008) = v8;
        *(_QWORD *)(v41 + 1016) = v10;
        v11 = 0;
        v9[62].m128i_i64[1] = 0;
        v9[63].m128i_i64[0] = 0;
        v9[63].m128i_i64[1] = 0;
      }
      else
      {
        v12 = v9[62].m128i_i64[1];
        v13 = v9[63].m128i_i64[1];
        v38 = v12;
        if ( v13 > 0 )
        {
          for ( i = 0; i != v13; ++i )
          {
            v15 = *(_QWORD *)(v12 + 32);
            if ( v15 )
            {
              sub_823A00(*(_QWORD *)v15, 16LL * (unsigned int)(*(_DWORD *)(v15 + 8) + 1), v6, v8, a5, a6);
              sub_823A00(v15, 16, v16, v17, v18, v19);
            }
            v12 += 40;
          }
        }
        v11 = v9[63].m128i_i64[0];
        if ( (const __m128i *)v38 == v9 )
          goto LABEL_10;
      }
      sub_823A00(v38, 40 * v11, v6, v8, a5, a6);
LABEL_10:
      ++v42;
      v9 = (const __m128i *)((char *)v9 + 1032);
      v41 += 1032;
    }
    while ( v42 != v37 );
  }
  if ( v35 == a1 + 8 )
    *(_DWORD *)(a1 + 4) = 0;
  else
    sub_823A00(v35, 1032 * v32, v6, v8, a5, a6);
LABEL_21:
  *(_QWORD *)(a1 + 3104) = v34;
  *(_QWORD *)(a1 + 3112) = v33;
  return a1;
}
