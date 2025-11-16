// Function: sub_1BFC440
// Address: 0x1bfc440
//
void __fastcall sub_1BFC440(_QWORD *a1, int a2)
{
  __int64 v2; // rax
  _QWORD *v3; // r12
  __int64 v4; // rax
  size_t v5; // r8
  size_t v6; // rdx
  void *v7; // r14
  __int64 v8; // rax
  _QWORD *v9; // r12
  __int64 v10; // rax
  size_t v11; // r8
  size_t v12; // rdx
  void *v13; // r14
  __int64 v14; // rax
  _QWORD *v15; // r12
  __int64 v16; // rax
  size_t v17; // r8
  size_t v18; // rdx
  void *v19; // r14
  __int64 v20; // rax
  __int64 v21; // r12
  unsigned int v22; // ebx
  __int64 v23; // rax
  size_t v24; // r8
  char *v25; // r15
  int v26; // ecx
  unsigned int v27; // ebx
  int v28; // ecx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  size_t v33; // [rsp+0h] [rbp-40h]
  size_t v34; // [rsp+0h] [rbp-40h]
  size_t v35; // [rsp+0h] [rbp-40h]
  size_t n; // [rsp+8h] [rbp-38h]
  size_t na; // [rsp+8h] [rbp-38h]
  size_t nb; // [rsp+8h] [rbp-38h]
  size_t nc; // [rsp+8h] [rbp-38h]
  size_t nd; // [rsp+8h] [rbp-38h]
  size_t ne; // [rsp+8h] [rbp-38h]
  size_t nf; // [rsp+8h] [rbp-38h]
  size_t ng; // [rsp+8h] [rbp-38h]

  v2 = sub_22077B0(24);
  v3 = (_QWORD *)v2;
  if ( v2 )
  {
    *(_QWORD *)v2 = 0;
    *(_QWORD *)(v2 + 8) = 0;
    *(_DWORD *)(v2 + 16) = a2;
    n = 8LL * ((unsigned int)(a2 + 63) >> 6);
    v4 = malloc(n);
    v5 = n;
    v6 = (unsigned int)(a2 + 63) >> 6;
    v7 = (void *)v4;
    if ( !v4 )
    {
      if ( n || (v32 = malloc(1u), v6 = (unsigned int)(a2 + 63) >> 6, v5 = 0, !v32) )
      {
        v35 = v5;
        nf = v6;
        sub_16BD1C0("Allocation failed", 1u);
        v6 = nf;
        v5 = v35;
      }
      else
      {
        v7 = (void *)v32;
      }
    }
    *v3 = v7;
    v3[1] = v6;
    if ( (unsigned int)(a2 + 63) >> 6 )
      memset(v7, 0, v5);
  }
  a1[3] = v3;
  v8 = sub_22077B0(24);
  v9 = (_QWORD *)v8;
  if ( v8 )
  {
    *(_QWORD *)v8 = 0;
    *(_QWORD *)(v8 + 8) = 0;
    *(_DWORD *)(v8 + 16) = a2;
    na = 8LL * ((unsigned int)(a2 + 63) >> 6);
    v10 = malloc(na);
    v11 = na;
    v12 = (unsigned int)(a2 + 63) >> 6;
    v13 = (void *)v10;
    if ( !v10 )
    {
      if ( na || (v31 = malloc(1u), v12 = (unsigned int)(a2 + 63) >> 6, v11 = 0, !v31) )
      {
        v34 = v11;
        ne = v12;
        sub_16BD1C0("Allocation failed", 1u);
        v12 = ne;
        v11 = v34;
      }
      else
      {
        v13 = (void *)v31;
      }
    }
    *v9 = v13;
    v9[1] = v12;
    if ( (unsigned int)(a2 + 63) >> 6 )
      memset(v13, 0, v11);
  }
  *a1 = v9;
  v14 = sub_22077B0(24);
  v15 = (_QWORD *)v14;
  if ( v14 )
  {
    *(_QWORD *)v14 = 0;
    *(_QWORD *)(v14 + 8) = 0;
    *(_DWORD *)(v14 + 16) = a2;
    nb = 8LL * ((unsigned int)(a2 + 63) >> 6);
    v16 = malloc(nb);
    v17 = nb;
    v18 = (unsigned int)(a2 + 63) >> 6;
    v19 = (void *)v16;
    if ( !v16 )
    {
      if ( nb || (v29 = malloc(1u), v18 = (unsigned int)(a2 + 63) >> 6, v17 = 0, !v29) )
      {
        v33 = v17;
        nd = v18;
        sub_16BD1C0("Allocation failed", 1u);
        v18 = nd;
        v17 = v33;
      }
      else
      {
        v19 = (void *)v29;
      }
    }
    *v15 = v19;
    v15[1] = v18;
    if ( (unsigned int)(a2 + 63) >> 6 )
      memset(v19, 0, v17);
  }
  a1[1] = v15;
  v20 = sub_22077B0(24);
  v21 = v20;
  if ( v20 )
  {
    *(_DWORD *)(v20 + 16) = a2;
    v22 = (unsigned int)(a2 + 63) >> 6;
    *(_QWORD *)v20 = 0;
    *(_QWORD *)(v20 + 8) = 0;
    v23 = malloc(8LL * v22);
    v24 = v22;
    v25 = (char *)v23;
    if ( !v23 )
    {
      if ( 8LL * v22 || (v30 = malloc(1u), v24 = v22, !v30) )
      {
        ng = v24;
        sub_16BD1C0("Allocation failed", 1u);
        v24 = ng;
      }
      else
      {
        v25 = (char *)v30;
      }
    }
    *(_QWORD *)v21 = v25;
    *(_QWORD *)(v21 + 8) = v24;
    if ( v22 )
    {
      nc = v24;
      memset(v25, -1, 8LL * v22);
      v26 = *(_DWORD *)(v21 + 16);
      v27 = (unsigned int)(v26 + 63) >> 6;
      if ( nc > v27 )
      {
        memset(&v25[8 * v27], 0, 8 * (nc - v27));
        v26 = *(_DWORD *)(v21 + 16);
      }
    }
    else
    {
      v26 = *(_DWORD *)(v21 + 16);
      v27 = (unsigned int)(v26 + 63) >> 6;
    }
    v28 = v26 & 0x3F;
    if ( v28 )
      *(_QWORD *)(*(_QWORD *)v21 + 8LL * (v27 - 1)) &= ~(-1LL << v28);
  }
  a1[2] = v21;
}
