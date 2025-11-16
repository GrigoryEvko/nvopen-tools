// Function: sub_2E23D10
// Address: 0x2e23d10
//
__int64 __fastcall sub_2E23D10(_QWORD *a1, int a2, __int64 a3)
{
  unsigned __int64 v6; // rdi
  __int64 *v7; // r9
  __int64 *v8; // rax
  int v9; // ecx
  __int64 v10; // r14
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r13
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  char v18; // di
  __int64 v20; // rax
  __int64 v21; // r13
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  __int64 *v24; // r9
  __int64 *v25; // rax
  int v26; // ecx
  __int64 v27; // rax
  unsigned __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rdx
  _QWORD *v31; // rcx
  char v32; // di
  char v33; // al
  unsigned __int64 v34; // rdx
  size_t v35; // r8
  unsigned __int64 v36; // r14
  _QWORD *v37; // r9
  __int64 v38; // r10
  __int64 **v39; // rax
  _QWORD *v40; // rdx
  void *v41; // rax
  _QWORD *v42; // rax
  _QWORD *v43; // rsi
  unsigned __int64 v44; // rdi
  _QWORD *v45; // rcx
  unsigned __int64 v46; // rdx
  _QWORD **v47; // rax
  unsigned __int64 v48; // rdi
  __int64 v49; // rdx
  _QWORD *v50; // [rsp+0h] [rbp-40h]
  size_t v51; // [rsp+0h] [rbp-40h]
  _QWORD *v52; // [rsp+0h] [rbp-40h]
  size_t nb; // [rsp+8h] [rbp-38h]
  size_t n; // [rsp+8h] [rbp-38h]
  size_t nc; // [rsp+8h] [rbp-38h]
  __int64 nd; // [rsp+8h] [rbp-38h]
  size_t na; // [rsp+8h] [rbp-38h]
  size_t ne; // [rsp+8h] [rbp-38h]

  v6 = a1[14];
  v7 = *(__int64 **)(a1[13] + 8 * (a2 % v6));
  if ( !v7 )
    goto LABEL_22;
  v8 = (__int64 *)*v7;
  if ( a2 != *(_DWORD *)(*v7 + 8) )
  {
    while ( *v8 )
    {
      v9 = *(_DWORD *)(*v8 + 8);
      v7 = v8;
      if ( a2 % v6 != v9 % v6 )
        break;
      v8 = (__int64 *)*v8;
      if ( a2 == v9 )
        goto LABEL_6;
    }
LABEL_22:
    n = a2;
    v20 = sub_22077B0(0x88u);
    v21 = v20;
    if ( v20 )
      *(_QWORD *)v20 = 0;
    *(_DWORD *)(v20 + 8) = a2;
    v22 = a1[14];
    *(_QWORD *)(v20 + 16) = v20 + 32;
    *(_QWORD *)(v20 + 24) = 0x200000000LL;
    *(_QWORD *)(v20 + 88) = 0x200000000LL;
    *(_QWORD *)(v20 + 80) = v20 + 96;
    *(_DWORD *)(v20 + 128) = a2 + 0x40000000;
    v23 = a1[13];
    *(_QWORD *)(v21 + 112) = 0;
    *(_QWORD *)(v21 + 120) = 0;
    *(_DWORD *)(v21 + 132) = 0;
    v24 = *(__int64 **)(v23 + 8 * (n % v22));
    if ( v24 )
    {
      v25 = (__int64 *)*v24;
      if ( *(_DWORD *)(*v24 + 8) == a2 )
      {
LABEL_29:
        v10 = *v24;
        if ( *v24 )
        {
          sub_2E22AE0((_QWORD *)v21);
LABEL_31:
          v27 = sub_22077B0(0x30u);
          *(_DWORD *)(v27 + 32) = a2;
          v28 = v27;
          *(_QWORD *)(v27 + 40) = a3;
          v29 = sub_2E23B70((__int64)(a1 + 20), (int *)(v27 + 32));
          if ( v30 )
          {
            v31 = a1 + 21;
            v32 = 1;
            if ( !v29 && (_QWORD *)v30 != v31 )
              v32 = *(_DWORD *)(v30 + 32) > a2;
            sub_220F040(v32, v28, (_QWORD *)v30, v31);
            ++a1[25];
          }
          else
          {
            j_j___libc_free_0(v28);
          }
          return v10 + 16;
        }
      }
      else
      {
        while ( *v25 )
        {
          v26 = *(_DWORD *)(*v25 + 8);
          v24 = v25;
          if ( n % v22 != v26 % v22 )
            break;
          v25 = (__int64 *)*v25;
          if ( v26 == a2 )
            goto LABEL_29;
        }
      }
    }
    v33 = sub_222DA10((__int64)(a1 + 17), v22, a1[16], 1);
    v35 = n;
    v36 = v34;
    if ( v33 )
    {
      if ( v34 == 1 )
      {
        v37 = a1 + 19;
        a1[19] = 0;
        na = (size_t)(a1 + 19);
      }
      else
      {
        if ( v34 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(a1 + 17, v22, v34);
        v51 = n;
        nd = 8 * v34;
        v41 = (void *)sub_22077B0(8 * v34);
        v42 = memset(v41, 0, nd);
        v35 = v51;
        v37 = v42;
        na = (size_t)(a1 + 19);
      }
      v43 = (_QWORD *)a1[15];
      a1[15] = 0;
      if ( v43 )
      {
        v44 = 0;
        do
        {
          v45 = v43;
          v43 = (_QWORD *)*v43;
          v46 = *((int *)v45 + 2) % v36;
          v47 = (_QWORD **)&v37[v46];
          if ( *v47 )
          {
            *v45 = **v47;
            **v47 = v45;
          }
          else
          {
            *v45 = a1[15];
            a1[15] = v45;
            *v47 = a1 + 15;
            if ( *v45 )
              v37[v44] = v45;
            v44 = v46;
          }
        }
        while ( v43 );
      }
      v48 = a1[13];
      if ( na != v48 )
      {
        v52 = v37;
        ne = v35;
        j_j___libc_free_0(v48);
        v37 = v52;
        v35 = ne;
      }
      a1[14] = v36;
      a1[13] = v37;
      v38 = v35 % v36;
    }
    else
    {
      v37 = (_QWORD *)a1[13];
      v38 = n % v22;
    }
    v39 = (__int64 **)&v37[v38];
    v40 = (_QWORD *)v37[v38];
    if ( v40 )
    {
      *(_QWORD *)v21 = *v40;
      **v39 = v21;
    }
    else
    {
      v49 = a1[15];
      a1[15] = v21;
      *(_QWORD *)v21 = v49;
      if ( v49 )
      {
        v37[(unsigned __int64)*(int *)(v49 + 8) % a1[14]] = v21;
        v39 = (__int64 **)(v38 * 8 + a1[13]);
      }
      *v39 = a1 + 15;
    }
    ++a1[16];
    v10 = v21;
    goto LABEL_31;
  }
LABEL_6:
  v10 = *v7;
  if ( !*v7 )
    goto LABEL_22;
  v11 = a1[22];
  v12 = (unsigned __int64)(a1 + 21);
  if ( !v11 )
    goto LABEL_14;
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 16);
      v14 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) >= a2 )
        break;
      v11 = *(_QWORD *)(v11 + 24);
      if ( !v14 )
        goto LABEL_12;
    }
    v12 = v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v13 );
LABEL_12:
  if ( (_QWORD *)v12 == a1 + 21 || *(_DWORD *)(v12 + 32) > a2 )
  {
LABEL_14:
    nb = v12;
    v50 = a1 + 21;
    v15 = sub_22077B0(0x30u);
    *(_DWORD *)(v15 + 32) = a2;
    v12 = v15;
    *(_QWORD *)(v15 + 40) = 0;
    v16 = sub_2E23C10(a1 + 20, nb, (int *)(v15 + 32));
    if ( v17 )
    {
      v18 = v16 || v50 == (_QWORD *)v17 || a2 < *(_DWORD *)(v17 + 32);
      sub_220F040(v18, v12, (_QWORD *)v17, v50);
      ++a1[25];
    }
    else
    {
      nc = v16;
      j_j___libc_free_0(v12);
      v12 = nc;
    }
  }
  *(_QWORD *)(v12 + 40) = sub_2FF6970(*a1, *(_QWORD *)(v12 + 40), a3, v13);
  return v10 + 16;
}
