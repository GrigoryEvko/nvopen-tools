// Function: sub_20FAF80
// Address: 0x20faf80
//
__int64 __fastcall sub_20FAF80(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  unsigned __int8 *v6; // r9
  __int64 v7; // r14
  __int64 v9; // rax
  unsigned int *v10; // r8
  __int64 v11; // rax
  unsigned int *v12; // r8
  _QWORD *v13; // r13
  __m128i v14; // xmm0
  _QWORD *v15; // r10
  __int64 v16; // rax
  unsigned __int64 v17; // r12
  __int64 *v18; // rax
  __int64 v19; // r15
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rdi
  char v23; // al
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // r15
  _QWORD *v26; // r9
  unsigned __int64 v27; // r8
  __int64 v28; // r8
  _QWORD **v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  void *v34; // rax
  _QWORD *v35; // rax
  _QWORD *v36; // r10
  _QWORD *v37; // rsi
  unsigned __int64 v38; // rdi
  _QWORD *v39; // rcx
  unsigned __int64 v40; // rdx
  _QWORD **v41; // rax
  _QWORD *v42; // rdi
  __int64 v43; // rdx
  __int64 v44; // [rsp+0h] [rbp-60h]
  __int64 v45; // [rsp+8h] [rbp-58h]
  unsigned int *v46; // [rsp+10h] [rbp-50h]
  unsigned __int64 v47; // [rsp+10h] [rbp-50h]
  unsigned __int8 *n; // [rsp+18h] [rbp-48h]
  size_t na; // [rsp+18h] [rbp-48h]
  __int64 nb; // [rsp+18h] [rbp-48h]
  size_t nc; // [rsp+18h] [rbp-48h]
  __m128i v52; // [rsp+20h] [rbp-40h] BYREF

  v52.m128i_i64[0] = sub_15B1030(a2);
  n = (unsigned __int8 *)v52.m128i_i64[0];
  v52.m128i_i64[1] = a3;
  v5 = sub_20FAE40(a1 + 8, (unsigned __int64)(a3 + 31 * v52.m128i_i64[0]) % a1[9], &v52, a3 + 31 * v52.m128i_i64[0]);
  v6 = n;
  if ( v5 && *v5 )
    return *v5 + 24LL;
  if ( (unsigned int)*n - 18 > 1 )
  {
    if ( a3 )
    {
      v31 = *(unsigned int *)(a3 + 8);
      v32 = 0;
      if ( (_DWORD)v31 == 2 )
        v32 = *(_QWORD *)(a3 - 8);
      v33 = sub_20FB390(a1, *(_QWORD *)(a3 - 8 * v31), v32);
      v6 = n;
      v10 = (unsigned int *)v33;
    }
    else
    {
      v10 = 0;
    }
  }
  else
  {
    v9 = sub_20FAF80(a1, *(_QWORD *)&n[8 * (1LL - *((unsigned int *)n + 2))], a3);
    v6 = n;
    v10 = (unsigned int *)v9;
  }
  v46 = v10;
  na = (size_t)v6;
  v11 = sub_22077B0(216);
  v12 = v46;
  v13 = (_QWORD *)v11;
  if ( v11 )
    *(_QWORD *)v11 = 0;
  v14 = _mm_loadu_si128(&v52);
  *(_QWORD *)(v11 + 24) = v46;
  v45 = v11 + 72;
  v15 = (_QWORD *)(v11 + 8);
  v7 = v11 + 24;
  *(_QWORD *)(v11 + 56) = v11 + 72;
  *(_QWORD *)(v11 + 32) = na;
  *(_QWORD *)(v11 + 40) = a3;
  *(_BYTE *)(v11 + 48) = 0;
  *(_QWORD *)(v11 + 64) = 0x400000000LL;
  v44 = v11 + 120;
  *(_QWORD *)(v11 + 104) = v11 + 120;
  *(_QWORD *)(v11 + 112) = 0x400000000LL;
  *(_QWORD *)(v11 + 184) = 0;
  *(_QWORD *)(v11 + 192) = 0;
  *(_QWORD *)(v11 + 200) = 0;
  *(__m128i *)(v11 + 8) = v14;
  if ( v46 )
  {
    v16 = v46[10];
    if ( (unsigned int)v16 >= v46[11] )
    {
      sub_16CD150((__int64)(v46 + 8), v46 + 12, 0, 8, (int)v46, na);
      v12 = v46;
      v15 = v13 + 1;
      v16 = v46[10];
    }
    *(_QWORD *)(*((_QWORD *)v12 + 4) + 8 * v16) = v7;
    ++v12[10];
  }
  v47 = a1[9];
  v17 = v13[2] + 31LL * v13[1];
  v18 = sub_20FAE40(a1 + 8, v17 % v47, v15, v17);
  if ( v18 && (v19 = *v18) != 0 )
  {
    v20 = v13[13];
    if ( v44 != v20 )
      _libc_free(v20);
    v21 = v13[7];
    if ( v45 != v21 )
      _libc_free(v21);
    v7 = v19 + 24;
    j_j___libc_free_0(v13, 216);
  }
  else
  {
    v22 = a1 + 12;
    v23 = sub_222DA10(a1 + 12, v47, a1[11], 1);
    v25 = v24;
    if ( v23 )
    {
      if ( v24 == 1 )
      {
        v26 = a1 + 14;
        a1[14] = 0;
        v36 = a1 + 14;
      }
      else
      {
        if ( v24 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v22, v47, v24);
        nb = 8 * v24;
        v34 = (void *)sub_22077B0(8 * v24);
        v35 = memset(v34, 0, nb);
        v36 = a1 + 14;
        v26 = v35;
      }
      v37 = (_QWORD *)a1[10];
      a1[10] = 0;
      if ( v37 )
      {
        v38 = 0;
        do
        {
          v39 = v37;
          v37 = (_QWORD *)*v37;
          v40 = v39[26] % v25;
          v41 = (_QWORD **)&v26[v40];
          if ( *v41 )
          {
            *v39 = **v41;
            **v41 = v39;
          }
          else
          {
            *v39 = a1[10];
            a1[10] = v39;
            *v41 = a1 + 10;
            if ( *v39 )
              v26[v38] = v39;
            v38 = v40;
          }
        }
        while ( v37 );
      }
      v42 = (_QWORD *)a1[8];
      if ( v42 != v36 )
      {
        nc = (size_t)v26;
        j_j___libc_free_0(v42, 8LL * a1[9]);
        v26 = (_QWORD *)nc;
      }
      a1[9] = v25;
      a1[8] = v26;
      v27 = v17 % v25;
    }
    else
    {
      v26 = (_QWORD *)a1[8];
      v27 = v17 % v47;
    }
    v28 = v27;
    v13[26] = v17;
    v29 = (_QWORD **)&v26[v28];
    v30 = (_QWORD *)v26[v28];
    if ( v30 )
    {
      *v13 = *v30;
      **v29 = v13;
    }
    else
    {
      v43 = a1[10];
      a1[10] = v13;
      *v13 = v43;
      if ( v43 )
      {
        v26[*(_QWORD *)(v43 + 208) % a1[9]] = v13;
        v29 = (_QWORD **)(v28 * 8 + a1[8]);
      }
      *v29 = a1 + 10;
    }
    ++a1[11];
  }
  return v7;
}
