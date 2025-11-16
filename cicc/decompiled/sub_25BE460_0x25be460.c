// Function: sub_25BE460
// Address: 0x25be460
//
void __fastcall sub_25BE460(int *a1, __int64 a2)
{
  __int64 v2; // r13
  int v4; // eax
  unsigned int v5; // esi
  int v6; // r12d
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 *v9; // r10
  int v10; // r14d
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // rsi
  __int64 v15; // r13
  __int32 v16; // r15d
  __int64 v17; // rdx
  __m128i *v18; // r12
  int v19; // eax
  int v20; // edx
  unsigned __int64 v21; // r14
  __int8 *v22; // rsi
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rcx
  bool v25; // cf
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // r8
  __int64 v28; // rax
  __m128i *v29; // rcx
  unsigned __int64 v30; // r8
  unsigned __int64 v31; // rax
  char *v32; // rsi
  __m128i *v33; // rdx
  const __m128i *v34; // rax
  unsigned __int64 v35; // [rsp+0h] [rbp-60h]
  __int64 v36; // [rsp+8h] [rbp-58h]
  __m128i *v37; // [rsp+8h] [rbp-58h]
  unsigned __int64 v38; // [rsp+10h] [rbp-50h]
  unsigned __int64 v39; // [rsp+10h] [rbp-50h]
  __int64 v40[2]; // [rsp+18h] [rbp-48h] BYREF
  __int64 *v41; // [rsp+28h] [rbp-38h] BYREF

  v2 = (__int64)(a1 + 2);
  v4 = *a1;
  v40[0] = a2;
  v5 = a1[8];
  v6 = v4 + 1;
  *a1 = v4 + 1;
  if ( !v5 )
  {
    ++*((_QWORD *)a1 + 1);
    v41 = 0;
    goto LABEL_46;
  }
  v7 = *((_QWORD *)a1 + 2);
  v8 = v40[0];
  v9 = 0;
  v10 = 1;
  v11 = (v5 - 1) & ((LODWORD(v40[0]) >> 9) ^ (LODWORD(v40[0]) >> 4));
  v12 = (__int64 *)(v7 + 16LL * v11);
  v13 = *v12;
  if ( v40[0] != *v12 )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v9 )
        v9 = v12;
      v11 = (v5 - 1) & (v10 + v11);
      v12 = (__int64 *)(v7 + 16LL * v11);
      v13 = *v12;
      if ( v40[0] == *v12 )
        goto LABEL_3;
      ++v10;
    }
    if ( !v9 )
      v9 = v12;
    v19 = a1[6];
    ++*((_QWORD *)a1 + 1);
    v20 = v19 + 1;
    v41 = v9;
    if ( 4 * (v19 + 1) < 3 * v5 )
    {
      if ( v5 - a1[7] - v20 > v5 >> 3 )
        goto LABEL_21;
      goto LABEL_47;
    }
LABEL_46:
    v5 *= 2;
LABEL_47:
    sub_25BE280(v2, v5);
    sub_25BC7E0(v2, v40, &v41);
    v8 = v40[0];
    v9 = v41;
    v20 = a1[6] + 1;
LABEL_21:
    a1[6] = v20;
    if ( *v9 != -4096 )
      --a1[7];
    *v9 = v8;
    *((_DWORD *)v9 + 2) = 0;
    *((_DWORD *)v9 + 2) = v6;
    v14 = (_BYTE *)*((_QWORD *)a1 + 6);
    if ( v14 != *((_BYTE **)a1 + 7) )
      goto LABEL_4;
LABEL_24:
    sub_25BCF70((__int64)(a1 + 10), v14, v40);
    v15 = v40[0];
    goto LABEL_7;
  }
LABEL_3:
  *((_DWORD *)v12 + 2) = v6;
  v14 = (_BYTE *)*((_QWORD *)a1 + 6);
  if ( v14 == *((_BYTE **)a1 + 7) )
    goto LABEL_24;
LABEL_4:
  v15 = v40[0];
  if ( v14 )
  {
    *(_QWORD *)v14 = v40[0];
    v14 = (_BYTE *)*((_QWORD *)a1 + 6);
  }
  *((_QWORD *)a1 + 6) = v14 + 8;
LABEL_7:
  v16 = *a1;
  v17 = *(_QWORD *)(v15 + 16);
  v18 = (__m128i *)*((_QWORD *)a1 + 12);
  if ( v18 != *((__m128i **)a1 + 13) )
  {
    if ( v18 )
    {
      v18->m128i_i64[0] = v15;
      v18->m128i_i64[1] = v17;
      v18[1].m128i_i32[0] = v16;
      v18 = (__m128i *)*((_QWORD *)a1 + 12);
    }
    *((_QWORD *)a1 + 12) = (char *)v18 + 24;
    return;
  }
  v21 = *((_QWORD *)a1 + 11);
  v22 = &v18->m128i_i8[-v21];
  v23 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v18->m128i_i64 - v21) >> 3);
  if ( v23 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v24 = 1;
  if ( v23 )
    v24 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v18->m128i_i64 - v21) >> 3);
  v25 = __CFADD__(v24, v23);
  v26 = v24 - 0x5555555555555555LL * ((__int64)((__int64)v18->m128i_i64 - v21) >> 3);
  if ( v25 )
  {
    v27 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v26 )
    {
      v31 = 24;
      v30 = 0;
      v29 = 0;
      goto LABEL_34;
    }
    if ( v26 > 0x555555555555555LL )
      v26 = 0x555555555555555LL;
    v27 = 24 * v26;
  }
  v36 = *(_QWORD *)(v15 + 16);
  v38 = v27;
  v28 = sub_22077B0(v27);
  v17 = v36;
  v22 = &v18->m128i_i8[-v21];
  v29 = (__m128i *)v28;
  v30 = v28 + v38;
  v31 = v28 + 24;
LABEL_34:
  v32 = &v22[(_QWORD)v29];
  if ( v32 )
  {
    *(_QWORD *)v32 = v15;
    *((_QWORD *)v32 + 1) = v17;
    *((_DWORD *)v32 + 4) = v16;
  }
  if ( v18 != (__m128i *)v21 )
  {
    v33 = v29;
    v34 = (const __m128i *)v21;
    do
    {
      if ( v33 )
      {
        *v33 = _mm_loadu_si128(v34);
        v33[1].m128i_i64[0] = v34[1].m128i_i64[0];
      }
      v34 = (const __m128i *)((char *)v34 + 24);
      v33 = (__m128i *)((char *)v33 + 24);
    }
    while ( v18 != v34 );
    v31 = (unsigned __int64)&v29[3] + 8 * (((unsigned __int64)&v18[-2].m128i_u64[1] - v21) >> 3);
  }
  if ( v21 )
  {
    v35 = v30;
    v37 = v29;
    v39 = v31;
    j_j___libc_free_0(v21);
    v30 = v35;
    v29 = v37;
    v31 = v39;
  }
  *((_QWORD *)a1 + 11) = v29;
  *((_QWORD *)a1 + 12) = v31;
  *((_QWORD *)a1 + 13) = v30;
}
