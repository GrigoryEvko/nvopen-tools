// Function: sub_1DCA4E0
// Address: 0x1dca4e0
//
__int64 *__fastcall sub_1DCA4E0(_QWORD *a1, int a2, __int64 a3)
{
  int v3; // r15d
  unsigned __int64 v6; // rdi
  __int64 **v7; // r8
  __int64 *v8; // rax
  int v9; // ecx
  __int64 *v10; // r13
  __int64 v11; // rax
  _QWORD *v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rsi
  _QWORD *v17; // r15
  __int64 v18; // r8
  __int64 v19; // r9
  size_t v20; // r8
  __int64 v21; // r12
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v26; // rax
  __int64 *v27; // r12
  int v28; // r10d
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // r15
  unsigned __int64 v31; // r13
  __int64 **v32; // r11
  __int64 *v33; // rax
  int v34; // ecx
  __int64 v35; // r15
  _QWORD *v36; // r8
  _QWORD *v37; // r9
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  int v40; // r15d
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // rax
  __int64 v44; // rdx
  _QWORD *v45; // rcx
  _BOOL8 v46; // rdi
  char v47; // al
  unsigned __int64 v48; // rdx
  unsigned __int64 v49; // r13
  _QWORD *v50; // r8
  __int64 ***v51; // rax
  __int64 *v52; // rdx
  void *v53; // rax
  _QWORD *v54; // rax
  _QWORD *v55; // r10
  _QWORD *v56; // rsi
  unsigned __int64 v57; // rdi
  _QWORD *v58; // rcx
  unsigned __int64 v59; // rdx
  _QWORD **v60; // rax
  _QWORD *v61; // rdi
  __int64 v62; // rdx
  __int64 v63; // [rsp+8h] [rbp-58h]
  size_t na; // [rsp+10h] [rbp-50h]
  size_t n; // [rsp+10h] [rbp-50h]
  size_t nb; // [rsp+10h] [rbp-50h]
  __int64 nc; // [rsp+10h] [rbp-50h]
  size_t nd; // [rsp+10h] [rbp-50h]
  _DWORD v69[3]; // [rsp+1Ch] [rbp-44h] BYREF
  int *v70[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = a2;
  v6 = a1[44];
  v69[0] = a2;
  v7 = *(__int64 ***)(a1[43] + 8 * (a2 % v6));
  if ( v7 )
  {
    v8 = *v7;
    if ( a2 != *((_DWORD *)*v7 + 2) )
    {
      while ( *v8 )
      {
        v9 = *(_DWORD *)(*v8 + 8);
        v7 = (__int64 **)v8;
        if ( a2 % v6 != v9 % v6 )
          break;
        v8 = (__int64 *)*v8;
        if ( a2 == v9 )
          goto LABEL_6;
      }
      goto LABEL_27;
    }
LABEL_6:
    v10 = *v7;
    if ( *v7 )
    {
      v11 = a1[52];
      v12 = a1 + 51;
      v13 = (__int64)(a1 + 51);
      if ( !v11 )
        goto LABEL_14;
      v14 = a1[52];
      do
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(v14 + 16);
          v16 = *(_QWORD *)(v14 + 24);
          if ( v3 <= *(_DWORD *)(v14 + 32) )
            break;
          v14 = *(_QWORD *)(v14 + 24);
          if ( !v16 )
            goto LABEL_12;
        }
        v13 = v14;
        v14 = *(_QWORD *)(v14 + 16);
      }
      while ( v15 );
LABEL_12:
      if ( (_QWORD *)v13 != v12 && v3 >= *(_DWORD *)(v13 + 32) )
      {
        v19 = *(_QWORD *)(v13 + 40);
        v20 = a1[29];
      }
      else
      {
LABEL_14:
        v17 = a1 + 50;
        v70[0] = v69;
        v18 = sub_1DCA430(a1 + 50, v13, v70);
        v11 = a1[52];
        v12 = a1 + 51;
        v19 = *(_QWORD *)(v18 + 40);
        v20 = a1[29];
        if ( !v11 )
        {
          v21 = (__int64)(a1 + 51);
          goto LABEL_23;
        }
        v3 = v69[0];
      }
      v21 = (__int64)v12;
      do
      {
        while ( 1 )
        {
          v22 = *(_QWORD *)(v11 + 16);
          v23 = *(_QWORD *)(v11 + 24);
          if ( *(_DWORD *)(v11 + 32) >= v3 )
            break;
          v11 = *(_QWORD *)(v11 + 24);
          if ( !v23 )
            goto LABEL_20;
        }
        v21 = v11;
        v11 = *(_QWORD *)(v11 + 16);
      }
      while ( v22 );
LABEL_20:
      if ( v12 != (_QWORD *)v21 && v3 >= *(_DWORD *)(v21 + 32) )
        goto LABEL_24;
      v17 = a1 + 50;
LABEL_23:
      v63 = v19;
      na = v20;
      v70[0] = v69;
      v24 = sub_1DCA430(v17, v21, v70);
      v19 = v63;
      v20 = na;
      v21 = v24;
LABEL_24:
      *(_QWORD *)(v21 + 40) = sub_1F4AF90(v20, v19, a3, 255);
      return v10 + 2;
    }
  }
LABEL_27:
  v26 = sub_22077B0(136);
  v27 = (__int64 *)v26;
  if ( v26 )
    *(_QWORD *)v26 = 0;
  v28 = v69[0];
  v29 = a1[44];
  *(_QWORD *)(v26 + 16) = v26 + 32;
  *(_DWORD *)(v26 + 128) = v3 + 0x40000000;
  v30 = v28;
  *(_QWORD *)(v26 + 24) = 0x200000000LL;
  *(_QWORD *)(v26 + 88) = 0x200000000LL;
  *(_DWORD *)(v26 + 8) = v28;
  *(_QWORD *)(v26 + 80) = v26 + 96;
  *(_QWORD *)(v26 + 112) = 0;
  *(_QWORD *)(v26 + 120) = 0;
  *(_DWORD *)(v26 + 132) = 0;
  v31 = v28 % v29;
  n = v31;
  v32 = *(__int64 ***)(a1[43] + 8 * v31);
  if ( !v32 )
    goto LABEL_45;
  v33 = *v32;
  if ( v28 != *((_DWORD *)*v32 + 2) )
  {
    while ( *v33 )
    {
      v34 = *(_DWORD *)(*v33 + 8);
      v32 = (__int64 **)v33;
      if ( v31 != v34 % v29 )
        break;
      v33 = (__int64 *)*v33;
      if ( v28 == v34 )
        goto LABEL_34;
    }
    goto LABEL_45;
  }
LABEL_34:
  v10 = *v32;
  if ( !*v32 )
  {
LABEL_45:
    v47 = sub_222DA10(a1 + 47, v29, a1[46], 1);
    v49 = v48;
    if ( v47 )
    {
      if ( v48 == 1 )
      {
        v50 = a1 + 49;
        a1[49] = 0;
        v55 = a1 + 49;
      }
      else
      {
        if ( v48 > 0xFFFFFFFFFFFFFFFLL )
          sub_4261EA(a1 + 47, v29, v48);
        nc = 8 * v48;
        v53 = (void *)sub_22077B0(8 * v48);
        v54 = memset(v53, 0, nc);
        v55 = a1 + 49;
        v50 = v54;
      }
      v56 = (_QWORD *)a1[45];
      a1[45] = 0;
      if ( v56 )
      {
        v57 = 0;
        do
        {
          v58 = v56;
          v56 = (_QWORD *)*v56;
          v59 = *((int *)v58 + 2) % v49;
          v60 = (_QWORD **)&v50[v59];
          if ( *v60 )
          {
            *v58 = **v60;
            **v60 = v58;
          }
          else
          {
            *v58 = a1[45];
            a1[45] = v58;
            *v60 = a1 + 45;
            if ( *v58 )
              v50[v57] = v58;
            v57 = v59;
          }
        }
        while ( v56 );
      }
      v61 = (_QWORD *)a1[43];
      if ( v61 != v55 )
      {
        nd = (size_t)v50;
        j_j___libc_free_0(v61, 8LL * a1[44]);
        v50 = (_QWORD *)nd;
      }
      a1[44] = v49;
      a1[43] = v50;
      n = v30 % v49;
    }
    else
    {
      v50 = (_QWORD *)a1[43];
    }
    v51 = (__int64 ***)&v50[n];
    v52 = (__int64 *)v50[n];
    if ( v52 )
    {
      *v27 = *v52;
      **v51 = v27;
    }
    else
    {
      v62 = a1[45];
      a1[45] = v27;
      *v27 = v62;
      if ( v62 )
      {
        v50[(unsigned __int64)*(int *)(v62 + 8) % a1[44]] = v27;
        v51 = (__int64 ***)(a1[43] + n * 8);
      }
      *v51 = (__int64 **)(a1 + 45);
    }
    ++a1[46];
    v10 = v27;
    goto LABEL_42;
  }
  sub_1DB4CE0((__int64)(v27 + 2));
  v35 = v27[14];
  v36 = v27 + 4;
  v37 = v27 + 12;
  if ( v35 )
  {
    sub_1DC95B0(*(_QWORD *)(v35 + 16));
    j_j___libc_free_0(v35, 48);
    v37 = v27 + 12;
    v36 = v27 + 4;
  }
  v38 = v27[10];
  if ( v37 != (_QWORD *)v38 )
  {
    nb = (size_t)v36;
    _libc_free(v38);
    v36 = (_QWORD *)nb;
  }
  v39 = v27[2];
  if ( v36 != (_QWORD *)v39 )
    _libc_free(v39);
  j_j___libc_free_0(v27, 136);
LABEL_42:
  v40 = v69[0];
  v41 = sub_22077B0(48);
  *(_DWORD *)(v41 + 32) = v69[0];
  v42 = v41;
  *(_QWORD *)(v41 + 40) = a3;
  v43 = sub_1DCA290((__int64)(a1 + 50), (int *)(v41 + 32));
  if ( v44 )
  {
    v45 = a1 + 51;
    v46 = 1;
    if ( !v43 && (_QWORD *)v44 != v45 )
      v46 = v40 < *(_DWORD *)(v44 + 32);
    sub_220F040(v46, v42, v44, v45);
    ++a1[55];
  }
  else
  {
    j_j___libc_free_0(v42, 48);
  }
  return v10 + 2;
}
