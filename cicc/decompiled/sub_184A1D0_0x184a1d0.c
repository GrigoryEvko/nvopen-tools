// Function: sub_184A1D0
// Address: 0x184a1d0
//
unsigned __int64 __fastcall sub_184A1D0(int *a1, __int64 a2)
{
  __int64 v2; // r8
  int v4; // eax
  unsigned int v5; // esi
  int v6; // ecx
  __int64 v7; // r10
  __int64 v8; // rdi
  unsigned int v9; // edx
  unsigned __int64 result; // rax
  __int64 v11; // r9
  _BYTE *v12; // rsi
  __int64 v13; // r13
  __int32 v14; // r15d
  __int64 v15; // rdx
  __m128i *v16; // r12
  const __m128i *v17; // r14
  signed __int64 v18; // rsi
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rcx
  bool v21; // cf
  unsigned __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // rax
  __m128i *v25; // rcx
  __int64 v26; // r8
  __int8 *v27; // rsi
  __m128i *v28; // rdx
  const __m128i *v29; // rax
  int v30; // r13d
  unsigned __int64 v31; // r12
  int v32; // ecx
  int v33; // ecx
  int v34; // eax
  int v35; // esi
  __int64 v36; // r9
  unsigned int v37; // edx
  int v38; // r11d
  unsigned __int64 v39; // r10
  int v40; // eax
  int v41; // esi
  __int64 v42; // r9
  unsigned __int64 v43; // r10
  int v44; // r11d
  unsigned int v45; // edx
  __int64 v46; // r8
  __m128i *v47; // [rsp+0h] [rbp-50h]
  __int64 v48; // [rsp+8h] [rbp-48h]
  unsigned __int64 v49; // [rsp+8h] [rbp-48h]
  __int64 v50; // [rsp+10h] [rbp-40h]
  __int64 v51; // [rsp+10h] [rbp-40h]
  __int64 v52[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = (__int64)(a1 + 2);
  v4 = *a1;
  v52[0] = a2;
  v5 = a1[8];
  v6 = v4 + 1;
  *a1 = v4 + 1;
  if ( !v5 )
  {
    ++*((_QWORD *)a1 + 1);
    goto LABEL_43;
  }
  v7 = *((_QWORD *)a1 + 2);
  v8 = v52[0];
  v9 = (v5 - 1) & ((LODWORD(v52[0]) >> 9) ^ (LODWORD(v52[0]) >> 4));
  result = v7 + 16LL * v9;
  v11 = *(_QWORD *)result;
  if ( v52[0] == *(_QWORD *)result )
    goto LABEL_3;
  v30 = 1;
  v31 = 0;
  while ( v11 != -8 )
  {
    if ( v11 == -16 && !v31 )
      v31 = result;
    v9 = (v5 - 1) & (v30 + v9);
    result = v7 + 16LL * v9;
    v11 = *(_QWORD *)result;
    if ( v52[0] == *(_QWORD *)result )
      goto LABEL_3;
    ++v30;
  }
  v32 = a1[6];
  if ( v31 )
    result = v31;
  ++*((_QWORD *)a1 + 1);
  v33 = v32 + 1;
  if ( 4 * v33 >= 3 * v5 )
  {
LABEL_43:
    sub_1849300(v2, 2 * v5);
    v34 = a1[8];
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *((_QWORD *)a1 + 2);
      v33 = a1[6] + 1;
      v37 = (v34 - 1) & ((LODWORD(v52[0]) >> 9) ^ (LODWORD(v52[0]) >> 4));
      result = v36 + 16LL * v37;
      v8 = *(_QWORD *)result;
      if ( v52[0] != *(_QWORD *)result )
      {
        v38 = 1;
        v39 = 0;
        while ( v8 != -8 )
        {
          if ( !v39 && v8 == -16 )
            v39 = result;
          v37 = v35 & (v38 + v37);
          result = v36 + 16LL * v37;
          v8 = *(_QWORD *)result;
          if ( v52[0] == *(_QWORD *)result )
            goto LABEL_38;
          ++v38;
        }
        v8 = v52[0];
        if ( v39 )
          result = v39;
      }
      goto LABEL_38;
    }
    goto LABEL_73;
  }
  if ( v5 - a1[7] - v33 <= v5 >> 3 )
  {
    sub_1849300(v2, v5);
    v40 = a1[8];
    if ( v40 )
    {
      v8 = v52[0];
      v41 = v40 - 1;
      v42 = *((_QWORD *)a1 + 2);
      v43 = 0;
      v44 = 1;
      v33 = a1[6] + 1;
      v45 = (v40 - 1) & ((LODWORD(v52[0]) >> 9) ^ (LODWORD(v52[0]) >> 4));
      result = v42 + 16LL * v45;
      v46 = *(_QWORD *)result;
      if ( *(_QWORD *)result != v52[0] )
      {
        while ( v46 != -8 )
        {
          if ( !v43 && v46 == -16 )
            v43 = result;
          v45 = v41 & (v44 + v45);
          result = v42 + 16LL * v45;
          v46 = *(_QWORD *)result;
          if ( v52[0] == *(_QWORD *)result )
            goto LABEL_38;
          ++v44;
        }
        if ( v43 )
          result = v43;
      }
      goto LABEL_38;
    }
LABEL_73:
    ++a1[6];
    BUG();
  }
LABEL_38:
  a1[6] = v33;
  if ( *(_QWORD *)result != -8 )
    --a1[7];
  *(_QWORD *)result = v8;
  *(_DWORD *)(result + 8) = 0;
  v6 = *a1;
LABEL_3:
  *(_DWORD *)(result + 8) = v6;
  v12 = (_BYTE *)*((_QWORD *)a1 + 6);
  if ( v12 == *((_BYTE **)a1 + 7) )
  {
    result = (unsigned __int64)sub_18483E0((__int64)(a1 + 10), v12, v52);
    v13 = v52[0];
  }
  else
  {
    v13 = v52[0];
    if ( v12 )
    {
      *(_QWORD *)v12 = v52[0];
      v12 = (_BYTE *)*((_QWORD *)a1 + 6);
      v13 = v52[0];
    }
    *((_QWORD *)a1 + 6) = v12 + 8;
  }
  v14 = *a1;
  v15 = *(_QWORD *)(v13 + 8);
  v16 = (__m128i *)*((_QWORD *)a1 + 12);
  if ( v16 != *((__m128i **)a1 + 13) )
  {
    if ( v16 )
    {
      v16->m128i_i64[0] = v13;
      v16->m128i_i64[1] = v15;
      v16[1].m128i_i32[0] = v14;
      v16 = (__m128i *)*((_QWORD *)a1 + 12);
    }
    *((_QWORD *)a1 + 12) = (char *)v16 + 24;
    return result;
  }
  v17 = (const __m128i *)*((_QWORD *)a1 + 11);
  v18 = (char *)v16 - (char *)v17;
  v19 = 0xAAAAAAAAAAAAAAABLL * (((char *)v16 - (char *)v17) >> 3);
  if ( v19 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v20 = 1;
  if ( v19 )
    v20 = 0xAAAAAAAAAAAAAAABLL * (((char *)v16 - (char *)v17) >> 3);
  v21 = __CFADD__(v20, v19);
  v22 = v20 - 0x5555555555555555LL * (((char *)v16 - (char *)v17) >> 3);
  if ( v21 )
  {
    v23 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v22 )
    {
      result = 24;
      v26 = 0;
      v25 = 0;
      goto LABEL_21;
    }
    if ( v22 > 0x555555555555555LL )
      v22 = 0x555555555555555LL;
    v23 = 24 * v22;
  }
  v48 = *(_QWORD *)(v13 + 8);
  v50 = v23;
  v24 = sub_22077B0(v23);
  v15 = v48;
  v18 = (char *)v16 - (char *)v17;
  v25 = (__m128i *)v24;
  v26 = v24 + v50;
  result = v24 + 24;
LABEL_21:
  v27 = &v25->m128i_i8[v18];
  if ( v27 )
  {
    *(_QWORD *)v27 = v13;
    *((_QWORD *)v27 + 1) = v15;
    *((_DWORD *)v27 + 4) = v14;
  }
  if ( v16 != v17 )
  {
    v28 = v25;
    v29 = v17;
    do
    {
      if ( v28 )
      {
        *v28 = _mm_loadu_si128(v29);
        v28[1].m128i_i64[0] = v29[1].m128i_i64[0];
      }
      v29 = (const __m128i *)((char *)v29 + 24);
      v28 = (__m128i *)((char *)v28 + 24);
    }
    while ( v16 != v29 );
    result = (unsigned __int64)&v25[3] + 8 * ((unsigned __int64)((char *)&v16[-2].m128i_u64[1] - (char *)v17) >> 3);
  }
  if ( v17 )
  {
    v47 = v25;
    v49 = result;
    v51 = v26;
    j_j___libc_free_0(v17, *((_QWORD *)a1 + 13) - (_QWORD)v17);
    v25 = v47;
    result = v49;
    v26 = v51;
  }
  *((_QWORD *)a1 + 11) = v25;
  *((_QWORD *)a1 + 12) = result;
  *((_QWORD *)a1 + 13) = v26;
  return result;
}
