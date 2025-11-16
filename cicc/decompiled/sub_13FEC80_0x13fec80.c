// Function: sub_13FEC80
// Address: 0x13fec80
//
__int64 __fastcall sub_13FEC80(const __m128i **a1)
{
  const __m128i *v2; // rbx
  __int64 v3; // rdi
  __int64 result; // rax
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // rcx
  __int64 v9; // r12
  int v10; // eax
  _QWORD *v11; // rdx
  __int64 v12; // rdi
  int v13; // esi
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // r8
  _QWORD *v17; // rax
  unsigned int v18; // esi
  __int64 v19; // r10
  __int64 v20; // r8
  int v21; // r11d
  __int64 *v22; // rdx
  unsigned int v23; // edi
  __int64 *v24; // rax
  __int64 v25; // rcx
  int v26; // r9d
  int v27; // r9d
  __int64 v28; // r10
  unsigned int v29; // eax
  int v30; // ecx
  __int64 v31; // r8
  int v32; // edi
  __int64 *v33; // rsi
  int v34; // eax
  int v35; // eax
  __int64 v36; // rax
  __m128i *v37; // rsi
  int v38; // edi
  int v39; // edi
  __int64 v40; // r8
  __int64 *v41; // r9
  __int64 v42; // r15
  int v43; // eax
  __int64 v44; // rsi
  int v45; // r9d
  __m128i v46; // [rsp+10h] [rbp-50h] BYREF
  __int64 v47; // [rsp+20h] [rbp-40h]

  v2 = a1[2];
  while ( 1 )
  {
    v3 = sub_157EBA0(v2[-2].m128i_i64[1]);
    result = 0;
    if ( v3 )
    {
      result = sub_15F4D60(v3);
      v2 = a1[2];
    }
    v5 = v2[-1].m128i_u32[2];
    if ( (_DWORD)v5 == (_DWORD)result )
      return result;
    v6 = v2[-1].m128i_i64[0];
    v2[-1].m128i_i32[2] = v5 + 1;
    v7 = sub_15F4DF0(v6, v5);
    v8 = (*a1)->m128i_i64[1];
    v9 = (*a1)->m128i_i64[0];
    v10 = *(_DWORD *)(v8 + 24);
    v11 = *(_QWORD **)v9;
    if ( v10 )
    {
      v12 = *(_QWORD *)(v8 + 8);
      v13 = v10 - 1;
      v14 = (v10 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v7 == *v15 )
      {
LABEL_7:
        v17 = (_QWORD *)v15[1];
        if ( v11 != v17 )
        {
          while ( v17 )
          {
            v17 = (_QWORD *)*v17;
            if ( v11 == v17 )
              goto LABEL_10;
          }
          goto LABEL_12;
        }
LABEL_10:
        v18 = *(_DWORD *)(v9 + 32);
        v19 = v9 + 8;
        if ( !v18 )
          goto LABEL_15;
        goto LABEL_11;
      }
      v34 = 1;
      while ( v16 != -8 )
      {
        v45 = v34 + 1;
        v14 = v13 & (v34 + v14);
        v15 = (__int64 *)(v12 + 16LL * v14);
        v16 = *v15;
        if ( v7 == *v15 )
          goto LABEL_7;
        v34 = v45;
      }
      if ( !v11 )
        goto LABEL_10;
LABEL_12:
      v2 = a1[2];
    }
    else
    {
      if ( v11 )
        goto LABEL_12;
      v18 = *(_DWORD *)(v9 + 32);
      v19 = v9 + 8;
      if ( !v18 )
      {
LABEL_15:
        ++*(_QWORD *)(v9 + 8);
LABEL_16:
        sub_13FEAC0(v19, 2 * v18);
        v26 = *(_DWORD *)(v9 + 32);
        if ( !v26 )
          goto LABEL_63;
        v27 = v26 - 1;
        v28 = *(_QWORD *)(v9 + 16);
        v29 = v27 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v30 = *(_DWORD *)(v9 + 24) + 1;
        v22 = (__int64 *)(v28 + 16LL * v29);
        v31 = *v22;
        if ( v7 != *v22 )
        {
          v32 = 1;
          v33 = 0;
          while ( v31 != -8 )
          {
            if ( !v33 && v31 == -16 )
              v33 = v22;
            v29 = v27 & (v32 + v29);
            v22 = (__int64 *)(v28 + 16LL * v29);
            v31 = *v22;
            if ( v7 == *v22 )
              goto LABEL_37;
            ++v32;
          }
          if ( v33 )
            v22 = v33;
        }
        goto LABEL_37;
      }
LABEL_11:
      v20 = *(_QWORD *)(v9 + 16);
      v21 = 1;
      v22 = 0;
      v23 = (v18 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v24 = (__int64 *)(v20 + 16LL * v23);
      v25 = *v24;
      if ( v7 == *v24 )
        goto LABEL_12;
      while ( v25 != -8 )
      {
        if ( v25 != -16 || v22 )
          v24 = v22;
        v23 = (v18 - 1) & (v21 + v23);
        v25 = *(_QWORD *)(v20 + 16LL * v23);
        if ( v7 == v25 )
          goto LABEL_12;
        ++v21;
        v22 = v24;
        v24 = (__int64 *)(v20 + 16LL * v23);
      }
      if ( !v22 )
        v22 = v24;
      v35 = *(_DWORD *)(v9 + 24);
      ++*(_QWORD *)(v9 + 8);
      v30 = v35 + 1;
      if ( 4 * (v35 + 1) >= 3 * v18 )
        goto LABEL_16;
      if ( v18 - *(_DWORD *)(v9 + 28) - v30 <= v18 >> 3 )
      {
        sub_13FEAC0(v19, v18);
        v38 = *(_DWORD *)(v9 + 32);
        if ( !v38 )
        {
LABEL_63:
          ++*(_DWORD *)(v9 + 24);
          BUG();
        }
        v39 = v38 - 1;
        v40 = *(_QWORD *)(v9 + 16);
        v41 = 0;
        LODWORD(v42) = v39 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v30 = *(_DWORD *)(v9 + 24) + 1;
        v43 = 1;
        v22 = (__int64 *)(v40 + 16LL * (unsigned int)v42);
        v44 = *v22;
        if ( v7 != *v22 )
        {
          while ( v44 != -8 )
          {
            if ( v44 == -16 && !v41 )
              v41 = v22;
            v42 = v39 & (unsigned int)(v42 + v43);
            v22 = (__int64 *)(v40 + 16 * v42);
            v44 = *v22;
            if ( v7 == *v22 )
              goto LABEL_37;
            ++v43;
          }
          if ( v41 )
            v22 = v41;
        }
      }
LABEL_37:
      *(_DWORD *)(v9 + 24) = v30;
      if ( *v22 != -8 )
        --*(_DWORD *)(v9 + 28);
      *v22 = v7;
      *((_DWORD *)v22 + 2) = 0;
      v36 = sub_157EBA0(v7);
      v46.m128i_i64[0] = v7;
      v37 = (__m128i *)a1[2];
      v46.m128i_i64[1] = v36;
      LODWORD(v47) = 0;
      if ( v37 == a1[3] )
      {
        sub_13FDF40(a1 + 1, v37, &v46);
        v2 = a1[2];
      }
      else
      {
        if ( v37 )
        {
          *v37 = _mm_loadu_si128(&v46);
          v37[1].m128i_i64[0] = v47;
          v37 = (__m128i *)a1[2];
        }
        v2 = (__m128i *)((char *)v37 + 24);
        a1[2] = (__m128i *)((char *)v37 + 24);
      }
    }
  }
}
