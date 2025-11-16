// Function: sub_1E3B210
// Address: 0x1e3b210
//
__int64 *__fastcall sub_1E3B210(
        __int64 *a1,
        int *a2,
        int *a3,
        _DWORD *a4,
        _QWORD *a5,
        _QWORD *a6,
        __int64 *a7,
        __int64 *a8)
{
  __int64 v10; // rbx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rcx
  int *v14; // r14
  bool v15; // cf
  unsigned __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r15
  __int64 v19; // rax
  int v20; // edi
  int v21; // r11d
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // r13
  __int64 v27; // r15
  __int64 v28; // rax
  int v29; // eax
  unsigned int v30; // eax
  unsigned __int64 v31; // rbx
  __int64 v32; // rax
  size_t v33; // rdx
  void *v34; // r14
  __int64 v35; // rax
  int v36; // eax
  __int64 v37; // rbx
  size_t v38; // r14
  void *v39; // r8
  int v40; // eax
  int v41; // eax
  __int64 v42; // r12
  __int64 v43; // rax
  size_t v44; // rdx
  void *v45; // r13
  const void *v46; // rsi
  __int64 v47; // rax
  int v48; // eax
  __int64 v49; // r12
  __int64 v50; // rax
  size_t v51; // rdx
  void *v52; // r13
  const void *v53; // rsi
  int v54; // eax
  __int64 i; // r12
  unsigned __int64 v56; // rdi
  __int64 v58; // r15
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  _QWORD *v64; // [rsp+0h] [rbp-70h]
  int *v65; // [rsp+8h] [rbp-68h]
  _QWORD *v66; // [rsp+8h] [rbp-68h]
  __int64 v67; // [rsp+10h] [rbp-60h]
  _DWORD *v68; // [rsp+10h] [rbp-60h]
  __int64 v69; // [rsp+18h] [rbp-58h]
  __int64 v71; // [rsp+28h] [rbp-48h]
  size_t n; // [rsp+30h] [rbp-40h]
  size_t na; // [rsp+30h] [rbp-40h]
  size_t nb; // [rsp+30h] [rbp-40h]
  size_t nc; // [rsp+30h] [rbp-40h]
  size_t nd; // [rsp+30h] [rbp-40h]
  __int64 v77; // [rsp+38h] [rbp-38h]

  v10 = a1[1];
  v77 = *a1;
  v11 = 0xEEEEEEEEEEEEEEEFLL * ((v10 - *a1) >> 3);
  if ( v11 == 0x111111111111111LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v12 = 1;
  v14 = a2;
  if ( v11 )
    v12 = 0xEEEEEEEEEEEEEEEFLL * ((v10 - *a1) >> 3);
  v15 = __CFADD__(v12, v11);
  v16 = v12 - 0x1111111111111111LL * ((v10 - *a1) >> 3);
  v17 = (__int64)a2 - v77;
  if ( v15 )
  {
    v58 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v16 )
    {
      v69 = 0;
      v18 = 120;
      v71 = 0;
      goto LABEL_7;
    }
    if ( v16 > 0x111111111111111LL )
      v16 = 0x111111111111111LL;
    v58 = 120 * v16;
  }
  v64 = a6;
  v66 = a5;
  v68 = a4;
  v59 = sub_22077B0(v58);
  a4 = v68;
  a5 = v66;
  a6 = v64;
  v71 = v59;
  v69 = v59 + v58;
  v18 = v59 + 120;
LABEL_7:
  v19 = v71 + v17;
  if ( v71 + v17 )
  {
    v20 = *a4;
    *(_BYTE *)(v19 + 40) = 1;
    v21 = *a3;
    *(_QWORD *)(v19 + 48) = 0;
    v22 = *a7;
    *(_DWORD *)(v19 + 4) = v20;
    v23 = *a5;
    *(_DWORD *)v19 = v21;
    v24 = *a8;
    *(_QWORD *)(v19 + 24) = v22;
    *(_QWORD *)(v19 + 8) = v23;
    v25 = *a6;
    *(_DWORD *)(v19 + 36) = v24;
    *(_QWORD *)(v19 + 16) = v25;
    *(_QWORD *)(v19 + 56) = 0;
    *(_QWORD *)(v19 + 64) = 0;
    *(_DWORD *)(v19 + 72) = 0;
    *(_QWORD *)(v19 + 80) = 0;
    *(_QWORD *)(v19 + 88) = 0;
    *(_QWORD *)(v19 + 96) = 0;
    *(_DWORD *)(v19 + 104) = 0;
    *(_DWORD *)(v19 + 112) = 0;
  }
  v26 = v77;
  if ( a2 != (int *)v77 )
  {
    v67 = v10;
    v27 = v71;
    v65 = v14;
    while ( 1 )
    {
      if ( v27 )
      {
        *(_DWORD *)v27 = *(_DWORD *)v26;
        *(_DWORD *)(v27 + 4) = *(_DWORD *)(v26 + 4);
        *(_QWORD *)(v27 + 8) = *(_QWORD *)(v26 + 8);
        *(_QWORD *)(v27 + 16) = *(_QWORD *)(v26 + 16);
        *(_QWORD *)(v27 + 24) = *(_QWORD *)(v26 + 24);
        *(_DWORD *)(v27 + 32) = *(_DWORD *)(v26 + 32);
        *(_DWORD *)(v27 + 36) = *(_DWORD *)(v26 + 36);
        *(_BYTE *)(v27 + 40) = *(_BYTE *)(v26 + 40);
        *(_DWORD *)(v27 + 44) = *(_DWORD *)(v26 + 44);
        v28 = *(_QWORD *)(v26 + 48);
        *(_QWORD *)(v27 + 56) = 0;
        *(_QWORD *)(v27 + 48) = v28;
        *(_QWORD *)(v27 + 64) = 0;
        v29 = *(_DWORD *)(v26 + 72);
        *(_DWORD *)(v27 + 72) = v29;
        if ( v29 )
        {
          v30 = (unsigned int)(v29 + 63) >> 6;
          v31 = 8LL * v30;
          n = v30;
          v32 = malloc(v31);
          v33 = n;
          v34 = (void *)v32;
          if ( !v32 )
          {
            if ( v31 || (v62 = malloc(1u), v33 = n, !v62) )
            {
              na = v33;
              sub_16BD1C0("Allocation failed", 1u);
              v33 = na;
            }
            else
            {
              v34 = (void *)v62;
            }
          }
          *(_QWORD *)(v27 + 64) = v33;
          *(_QWORD *)(v27 + 56) = v34;
          memcpy(v34, *(const void **)(v26 + 56), v31);
        }
        v35 = *(_QWORD *)(v26 + 80);
        *(_QWORD *)(v27 + 88) = 0;
        *(_QWORD *)(v27 + 96) = 0;
        *(_QWORD *)(v27 + 80) = v35;
        v36 = *(_DWORD *)(v26 + 104);
        *(_DWORD *)(v27 + 104) = v36;
        if ( v36 )
        {
          v37 = (unsigned int)(v36 + 63) >> 6;
          v38 = 8 * v37;
          v39 = (void *)malloc(8 * v37);
          if ( !v39 )
          {
            if ( v38 || (v63 = malloc(1u), v39 = 0, !v63) )
            {
              nd = (size_t)v39;
              sub_16BD1C0("Allocation failed", 1u);
              v39 = (void *)nd;
            }
            else
            {
              v39 = (void *)v63;
            }
          }
          *(_QWORD *)(v27 + 88) = v39;
          *(_QWORD *)(v27 + 96) = v37;
          memcpy(v39, *(const void **)(v26 + 88), v38);
        }
        *(_DWORD *)(v27 + 112) = *(_DWORD *)(v26 + 112);
      }
      v26 += 120;
      if ( a2 == (int *)v26 )
        break;
      v27 += 120;
    }
    v10 = v67;
    v14 = v65;
    v18 = v27 + 240;
  }
  if ( a2 != (int *)v10 )
  {
    do
    {
      v40 = *v14;
      *(_QWORD *)(v18 + 56) = 0;
      *(_QWORD *)(v18 + 64) = 0;
      *(_DWORD *)v18 = v40;
      *(_DWORD *)(v18 + 4) = v14[1];
      *(_QWORD *)(v18 + 8) = *((_QWORD *)v14 + 1);
      *(_QWORD *)(v18 + 16) = *((_QWORD *)v14 + 2);
      *(_QWORD *)(v18 + 24) = *((_QWORD *)v14 + 3);
      *(_DWORD *)(v18 + 32) = v14[8];
      *(_DWORD *)(v18 + 36) = v14[9];
      *(_BYTE *)(v18 + 40) = *((_BYTE *)v14 + 40);
      *(_DWORD *)(v18 + 44) = v14[11];
      *(_QWORD *)(v18 + 48) = *((_QWORD *)v14 + 6);
      v41 = v14[18];
      *(_DWORD *)(v18 + 72) = v41;
      if ( v41 )
      {
        v42 = (unsigned int)(v41 + 63) >> 6;
        v43 = malloc(8 * v42);
        v44 = 8 * v42;
        v45 = (void *)v43;
        if ( !v43 )
        {
          if ( 8 * v42 || (v61 = malloc(1u), v44 = 0, !v61) )
          {
            nc = v44;
            sub_16BD1C0("Allocation failed", 1u);
            v44 = nc;
          }
          else
          {
            v45 = (void *)v61;
          }
        }
        *(_QWORD *)(v18 + 56) = v45;
        v46 = (const void *)*((_QWORD *)v14 + 7);
        *(_QWORD *)(v18 + 64) = v42;
        memcpy(v45, v46, v44);
      }
      v47 = *((_QWORD *)v14 + 10);
      *(_QWORD *)(v18 + 88) = 0;
      *(_QWORD *)(v18 + 96) = 0;
      *(_QWORD *)(v18 + 80) = v47;
      v48 = v14[26];
      *(_DWORD *)(v18 + 104) = v48;
      if ( v48 )
      {
        v49 = (unsigned int)(v48 + 63) >> 6;
        v50 = malloc(8 * v49);
        v51 = 8 * v49;
        v52 = (void *)v50;
        if ( !v50 )
        {
          if ( 8 * v49 || (v60 = malloc(1u), v51 = 0, !v60) )
          {
            nb = v51;
            sub_16BD1C0("Allocation failed", 1u);
            v51 = nb;
          }
          else
          {
            v52 = (void *)v60;
          }
        }
        *(_QWORD *)(v18 + 88) = v52;
        v53 = (const void *)*((_QWORD *)v14 + 11);
        *(_QWORD *)(v18 + 96) = v49;
        memcpy(v52, v53, v51);
      }
      v54 = v14[28];
      v14 += 30;
      v18 += 120;
      *(_DWORD *)(v18 - 8) = v54;
    }
    while ( (int *)v10 != v14 );
  }
  for ( i = v77; i != v10; _libc_free(*(_QWORD *)(i - 64)) )
  {
    v56 = *(_QWORD *)(i + 88);
    i += 120;
    _libc_free(v56);
  }
  if ( v77 )
    j_j___libc_free_0(v77, a1[2] - v77);
  *a1 = v71;
  a1[1] = v18;
  a1[2] = v69;
  return a1;
}
