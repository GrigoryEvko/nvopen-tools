// Function: sub_3118180
// Address: 0x3118180
//
__int64 __fastcall sub_3118180(__int64 a1, __int8 *a2, size_t a3)
{
  __int64 *v4; // r12
  int v6; // eax
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned int v10; // r15d
  __m128i *v12; // r9
  __m128i *v13; // rax
  unsigned int v14; // edx
  __int64 v15; // rax
  bool v16; // zf
  __m128i *v17; // rax
  __m128i *v18; // rdi
  __int64 v19; // rax
  const void *v20; // r13
  size_t v21; // r14
  int v22; // eax
  unsigned int v23; // r8d
  __int64 *v24; // r9
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned int v27; // r8d
  __int64 *v28; // r9
  __int64 v29; // rcx
  __int64 *v30; // rdx
  __m128i *v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rcx
  __m128i *v34; // rax
  __int64 v35; // rcx
  const __m128i *v36; // rax
  unsigned __int64 *v37; // r14
  __m128i *v38; // rdx
  const __m128i *v39; // rax
  __m128i *v40; // rcx
  const __m128i *v41; // rsi
  unsigned __int64 *v42; // r13
  __int64 v43; // r8
  int v44; // eax
  __int64 v45; // [rsp+8h] [rbp-78h]
  __m128i *v46; // [rsp+8h] [rbp-78h]
  __int64 *v47; // [rsp+10h] [rbp-70h]
  __m128i *v48; // [rsp+10h] [rbp-70h]
  __int64 v49; // [rsp+18h] [rbp-68h]
  unsigned int v50; // [rsp+18h] [rbp-68h]
  __m128i *v51; // [rsp+18h] [rbp-68h]
  int v52; // [rsp+18h] [rbp-68h]
  __int64 v53; // [rsp+28h] [rbp-58h] BYREF
  __m128i *v54; // [rsp+30h] [rbp-50h] BYREF
  __int64 v55; // [rsp+38h] [rbp-48h]
  __m128i v56[4]; // [rsp+40h] [rbp-40h] BYREF

  v4 = (__int64 *)(a1 + 80);
  v49 = a3;
  v6 = sub_C92610();
  v7 = sub_C92860((__int64 *)(a1 + 80), a2, a3, v6);
  if ( v7 == -1 || (v8 = *(_QWORD *)(a1 + 80), v9 = v8 + 8LL * v7, v9 == v8 + 8LL * *(unsigned int *)(a1 + 88)) )
  {
    v10 = *(_DWORD *)(a1 + 40);
    if ( !a2 )
    {
      v12 = v56;
      v55 = 0;
      v14 = v10;
      v54 = v56;
      v56[0].m128i_i8[0] = 0;
LABEL_11:
      if ( *(_DWORD *)(a1 + 44) <= v14 )
      {
        v32 = sub_C8D7D0(a1 + 32, a1 + 48, 0, 0x20u, (unsigned __int64 *)&v53, (__int64)v56);
        v12 = v56;
        v48 = (__m128i *)v32;
        v33 = 2LL * *(unsigned int *)(a1 + 40);
        v34 = (__m128i *)(v33 * 16 + v32);
        if ( v34 )
        {
          v34->m128i_i64[0] = (__int64)v34[1].m128i_i64;
          if ( v54 == v56 )
          {
            v34[1] = _mm_load_si128(v56);
          }
          else
          {
            v34->m128i_i64[0] = (__int64)v54;
            v34[1].m128i_i64[0] = v56[0].m128i_i64[0];
          }
          v54 = v56;
          v34->m128i_i64[1] = v55;
          v35 = *(unsigned int *)(a1 + 40);
          v55 = 0;
          v56[0].m128i_i8[0] = 0;
          v33 = 2 * v35;
        }
        v36 = *(const __m128i **)(a1 + 32);
        v37 = (unsigned __int64 *)&v36[v33];
        if ( v36 != &v36[v33] )
        {
          v38 = v48;
          v39 = v36 + 1;
          v40 = &v48[v33];
          do
          {
            if ( v38 )
            {
              v38->m128i_i64[0] = (__int64)v38[1].m128i_i64;
              v41 = (const __m128i *)v39[-1].m128i_i64[0];
              if ( v41 == v39 )
              {
                v38[1] = _mm_loadu_si128(v39);
              }
              else
              {
                v38->m128i_i64[0] = (__int64)v41;
                v38[1].m128i_i64[0] = v39->m128i_i64[0];
              }
              v38->m128i_i64[1] = v39[-1].m128i_i64[1];
              v39[-1].m128i_i64[0] = (__int64)v39;
              v39[-1].m128i_i64[1] = 0;
              v39->m128i_i8[0] = 0;
            }
            v38 += 2;
            v39 += 2;
          }
          while ( v38 != v40 );
          v42 = *(unsigned __int64 **)(a1 + 32);
          v43 = 4LL * *(unsigned int *)(a1 + 40);
          v37 = &v42[v43];
          if ( v42 != &v42[v43] )
          {
            do
            {
              v37 -= 4;
              if ( (unsigned __int64 *)*v37 != v37 + 2 )
              {
                v51 = v12;
                j_j___libc_free_0(*v37);
                v12 = v51;
              }
            }
            while ( v42 != v37 );
            v37 = *(unsigned __int64 **)(a1 + 32);
          }
        }
        v44 = v53;
        if ( (unsigned __int64 *)(a1 + 48) != v37 )
        {
          v46 = v12;
          v52 = v53;
          _libc_free((unsigned __int64)v37);
          v12 = v46;
          v44 = v52;
        }
        ++*(_DWORD *)(a1 + 40);
        *(_DWORD *)(a1 + 44) = v44;
        v18 = v54;
        *(_QWORD *)(a1 + 32) = v48;
      }
      else
      {
        v15 = 32LL * v14;
        v16 = *(_QWORD *)(a1 + 32) + v15 == 0;
        v17 = (__m128i *)(*(_QWORD *)(a1 + 32) + v15);
        v18 = v54;
        if ( !v16 )
        {
          v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
          if ( v54 == v56 )
          {
            v17[1] = _mm_load_si128(v56);
          }
          else
          {
            v17->m128i_i64[0] = (__int64)v54;
            v17[1].m128i_i64[0] = v56[0].m128i_i64[0];
          }
          v54 = v56;
          v12 = v56;
          v18 = v56;
          v17->m128i_i64[1] = v55;
          v14 = *(_DWORD *)(a1 + 40);
          v55 = 0;
          v56[0].m128i_i8[0] = 0;
        }
        *(_DWORD *)(a1 + 40) = v14 + 1;
      }
      if ( v18 != v12 )
        j_j___libc_free_0((unsigned __int64)v18);
      v19 = *(_QWORD *)(a1 + 32) + 32LL * *(unsigned int *)(a1 + 40) - 32;
      v20 = *(const void **)v19;
      v21 = *(_QWORD *)(v19 + 8);
      v22 = sub_C92610();
      v23 = sub_C92740((__int64)v4, v20, v21, v22);
      v24 = (__int64 *)(*(_QWORD *)(a1 + 80) + 8LL * v23);
      v25 = *v24;
      if ( *v24 )
      {
        if ( v25 != -8 )
        {
LABEL_21:
          *(_DWORD *)(v25 + 8) = v10;
          return v10;
        }
        --*(_DWORD *)(a1 + 96);
      }
      v47 = v24;
      v50 = v23;
      v26 = sub_C7D670(v21 + 17, 8);
      v27 = v50;
      v28 = v47;
      v29 = v26;
      if ( v21 )
      {
        v45 = v26;
        memcpy((void *)(v26 + 16), v20, v21);
        v27 = v50;
        v28 = v47;
        v29 = v45;
      }
      *(_BYTE *)(v29 + v21 + 16) = 0;
      *(_QWORD *)v29 = v21;
      *(_DWORD *)(v29 + 8) = 0;
      *v28 = v29;
      ++*(_DWORD *)(a1 + 92);
      v30 = (__int64 *)(*(_QWORD *)(a1 + 80) + 8LL * (unsigned int)sub_C929D0(v4, v27));
      v25 = *v30;
      if ( !*v30 || v25 == -8 )
      {
        do
        {
          do
          {
            v25 = v30[1];
            ++v30;
          }
          while ( v25 == -8 );
        }
        while ( !v25 );
      }
      goto LABEL_21;
    }
    v12 = v56;
    v53 = a3;
    v54 = v56;
    if ( a3 > 0xF )
    {
      v54 = (__m128i *)sub_22409D0((__int64)&v54, (unsigned __int64 *)&v53, 0);
      v31 = v54;
      v56[0].m128i_i64[0] = v53;
    }
    else
    {
      if ( a3 == 1 )
      {
        v56[0].m128i_i8[0] = *a2;
        v13 = v56;
LABEL_9:
        v55 = v49;
        v13->m128i_i8[v49] = 0;
        v14 = *(_DWORD *)(a1 + 40);
        goto LABEL_11;
      }
      if ( !a3 )
      {
        v13 = v56;
        goto LABEL_9;
      }
      v31 = v56;
    }
    memcpy(v31, a2, a3);
    v12 = v56;
    v49 = v53;
    v13 = v54;
    goto LABEL_9;
  }
  return *(unsigned int *)(*(_QWORD *)v9 + 8LL);
}
