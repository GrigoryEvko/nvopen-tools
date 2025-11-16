// Function: sub_3505300
// Address: 0x3505300
//
__int64 __fastcall sub_3505300(_QWORD *a1, _BYTE *a2, __int64 a3)
{
  _QWORD *v5; // rax
  unsigned __int8 *v6; // r9
  __int64 v7; // r14
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int *v12; // r8
  __int64 v13; // rax
  unsigned int *v14; // r8
  _QWORD *v15; // r13
  __m128i v16; // xmm0
  _QWORD *v17; // r10
  __int64 v18; // rax
  unsigned __int64 v19; // r12
  __int64 *v20; // rax
  __int64 v21; // r15
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  _QWORD *v24; // rdi
  char v25; // al
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // r15
  _QWORD *v28; // r9
  unsigned __int64 v29; // r8
  __int64 v30; // r8
  _QWORD **v31; // rax
  _QWORD *v32; // rdx
  unsigned __int8 v33; // al
  __int64 v34; // rdx
  _QWORD *v35; // rcx
  __int64 v36; // rax
  void *v37; // rax
  _QWORD *v38; // rax
  _QWORD *v39; // r10
  _QWORD *v40; // rsi
  unsigned __int64 v41; // rdi
  _QWORD *v42; // rcx
  unsigned __int64 v43; // rdx
  _QWORD **v44; // rax
  unsigned __int64 v45; // rdi
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // [rsp+0h] [rbp-60h]
  __int64 v49; // [rsp+8h] [rbp-58h]
  unsigned int *v50; // [rsp+10h] [rbp-50h]
  unsigned __int64 v51; // [rsp+10h] [rbp-50h]
  unsigned __int8 *n; // [rsp+18h] [rbp-48h]
  size_t na; // [rsp+18h] [rbp-48h]
  __int64 nb; // [rsp+18h] [rbp-48h]
  size_t nc; // [rsp+18h] [rbp-48h]
  __m128i v56; // [rsp+20h] [rbp-40h] BYREF

  v56.m128i_i64[0] = (__int64)sub_AF3520(a2);
  n = (unsigned __int8 *)v56.m128i_i64[0];
  v56.m128i_i64[1] = a3;
  v5 = sub_3505160(a1 + 8, (unsigned __int64)(a3 + 31 * v56.m128i_i64[0]) % a1[9], &v56, a3 + 31 * v56.m128i_i64[0]);
  v6 = n;
  if ( v5 && *v5 )
    return *v5 + 24LL;
  if ( (unsigned int)*n - 19 > 1 )
  {
    if ( a3 )
    {
      v33 = *(_BYTE *)(a3 - 16);
      if ( (v33 & 2) != 0 )
      {
        if ( *(_DWORD *)(a3 - 24) == 2 )
          v34 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + 8LL);
        else
          v34 = 0;
        v35 = *(_QWORD **)(a3 - 32);
      }
      else
      {
        v47 = a3 - 16;
        if ( ((*(_WORD *)(a3 - 16) >> 6) & 0xF) == 2 )
          v34 = *(_QWORD *)(v47 - 8LL * ((v33 >> 2) & 0xF) + 8);
        else
          v34 = 0;
        v35 = (_QWORD *)(v47 - 8LL * ((v33 >> 2) & 0xF));
      }
      v36 = sub_35057B0(a1, *v35, v34);
      v6 = n;
      v12 = (unsigned int *)v36;
    }
    else
    {
      v12 = 0;
    }
  }
  else
  {
    v9 = *(n - 16);
    if ( (v9 & 2) != 0 )
      v10 = *((_QWORD *)n - 4);
    else
      v10 = (__int64)&n[-8 * ((v9 >> 2) & 0xF) - 16];
    v11 = sub_3505300(a1, *(_QWORD *)(v10 + 8), a3);
    v6 = n;
    v12 = (unsigned int *)v11;
  }
  v50 = v12;
  na = (size_t)v6;
  v13 = sub_22077B0(0xD8u);
  v14 = v50;
  v15 = (_QWORD *)v13;
  if ( v13 )
    *(_QWORD *)v13 = 0;
  v16 = _mm_loadu_si128(&v56);
  *(_QWORD *)(v13 + 24) = v50;
  v49 = v13 + 72;
  v17 = (_QWORD *)(v13 + 8);
  v7 = v13 + 24;
  *(_QWORD *)(v13 + 56) = v13 + 72;
  *(_QWORD *)(v13 + 32) = na;
  *(_QWORD *)(v13 + 40) = a3;
  *(_BYTE *)(v13 + 48) = 0;
  *(_QWORD *)(v13 + 64) = 0x400000000LL;
  v48 = v13 + 120;
  *(_QWORD *)(v13 + 104) = v13 + 120;
  *(_QWORD *)(v13 + 112) = 0x400000000LL;
  *(_QWORD *)(v13 + 184) = 0;
  *(_QWORD *)(v13 + 192) = 0;
  *(_QWORD *)(v13 + 200) = 0;
  *(__m128i *)(v13 + 8) = v16;
  if ( v50 )
  {
    v18 = v50[10];
    if ( v18 + 1 > (unsigned __int64)v50[11] )
    {
      sub_C8D5F0((__int64)(v50 + 8), v50 + 12, v18 + 1, 8u, (__int64)v50, na);
      v14 = v50;
      v17 = v15 + 1;
      v18 = v50[10];
    }
    *(_QWORD *)(*((_QWORD *)v14 + 4) + 8 * v18) = v7;
    ++v14[10];
  }
  v51 = a1[9];
  v19 = v15[2] + 31LL * v15[1];
  v20 = sub_3505160(a1 + 8, v19 % v51, v17, v19);
  if ( v20 && (v21 = *v20) != 0 )
  {
    v22 = v15[13];
    if ( v48 != v22 )
      _libc_free(v22);
    v23 = v15[7];
    if ( v49 != v23 )
      _libc_free(v23);
    v7 = v21 + 24;
    j_j___libc_free_0((unsigned __int64)v15);
  }
  else
  {
    v24 = a1 + 12;
    v25 = sub_222DA10((__int64)(a1 + 12), v51, a1[11], 1);
    v27 = v26;
    if ( v25 )
    {
      if ( v26 == 1 )
      {
        v28 = a1 + 14;
        a1[14] = 0;
        v39 = a1 + 14;
      }
      else
      {
        if ( v26 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(v24, v51, v26);
        nb = 8 * v26;
        v37 = (void *)sub_22077B0(8 * v26);
        v38 = memset(v37, 0, nb);
        v39 = a1 + 14;
        v28 = v38;
      }
      v40 = (_QWORD *)a1[10];
      a1[10] = 0;
      if ( v40 )
      {
        v41 = 0;
        do
        {
          v42 = v40;
          v40 = (_QWORD *)*v40;
          v43 = v42[26] % v27;
          v44 = (_QWORD **)&v28[v43];
          if ( *v44 )
          {
            *v42 = **v44;
            **v44 = v42;
          }
          else
          {
            *v42 = a1[10];
            a1[10] = v42;
            *v44 = a1 + 10;
            if ( *v42 )
              v28[v41] = v42;
            v41 = v43;
          }
        }
        while ( v40 );
      }
      v45 = a1[8];
      if ( (_QWORD *)v45 != v39 )
      {
        nc = (size_t)v28;
        j_j___libc_free_0(v45);
        v28 = (_QWORD *)nc;
      }
      a1[9] = v27;
      a1[8] = v28;
      v29 = v19 % v27;
    }
    else
    {
      v28 = (_QWORD *)a1[8];
      v29 = v19 % v51;
    }
    v30 = v29;
    v15[26] = v19;
    v31 = (_QWORD **)&v28[v30];
    v32 = (_QWORD *)v28[v30];
    if ( v32 )
    {
      *v15 = *v32;
      **v31 = v15;
    }
    else
    {
      v46 = a1[10];
      a1[10] = v15;
      *v15 = v46;
      if ( v46 )
      {
        v28[*(_QWORD *)(v46 + 208) % a1[9]] = v15;
        v31 = (_QWORD **)(v30 * 8 + a1[8]);
      }
      *v31 = a1 + 10;
    }
    ++a1[11];
  }
  return v7;
}
