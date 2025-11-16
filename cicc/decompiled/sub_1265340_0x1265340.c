// Function: sub_1265340
// Address: 0x1265340
//
_QWORD *__fastcall sub_1265340(
        __int64 a1,
        size_t a2,
        __int64 *a3,
        __int64 a4,
        _DWORD *a5,
        char a6,
        int a7,
        char a8,
        char a9)
{
  int v10; // r12d
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // r13
  __int64 i; // rax
  __int64 v19; // r15
  __int64 v20; // r14
  size_t v21; // rdx
  __int64 v22; // rcx
  _BYTE *v23; // r8
  __int64 v24; // r9
  _QWORD *v25; // r12
  const char **v27; // rax
  const char *v28; // r13
  size_t v29; // rax
  __int64 v30; // r13
  __int64 v31; // rdi
  int v32; // r13d
  __int64 v33; // rax
  size_t v34; // rax
  const char *v35; // r14
  size_t v36; // rax
  __int64 v37; // rax
  __int64 v38; // r14
  __int64 v39; // rax
  size_t v40; // r15
  _QWORD *v41; // rax
  _BYTE *v42; // rdi
  _BYTE *v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rax
  _QWORD *v46; // rdi
  size_t v47; // [rsp+0h] [rbp-80h]
  const char *src; // [rsp+18h] [rbp-68h]
  _BYTE *srca; // [rsp+18h] [rbp-68h]
  size_t v51; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v52; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _QWORD v54[8]; // [rsp+40h] [rbp-40h] BYREF

  v47 = a2;
  if ( a6 )
  {
    sub_223E0D0(qword_4FD4BE0, "\"", 1);
    v27 = (const char **)a3[9];
    v28 = *v27;
    if ( *v27 )
    {
      v29 = strlen(*v27);
      sub_223E0D0(qword_4FD4BE0, v28, v29);
    }
    else
    {
      sub_222DC80(
        (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
        *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
    }
    sub_223E0D0(qword_4FD4BE0, "\" -lgenfe \"", 11);
    v30 = sub_223E0D0(qword_4FD4BE0, *a3, a3[1]);
    sub_223E0D0(v30, "\" -o \"", 6);
    v31 = v30;
    v32 = 1;
    v33 = sub_223E0D0(v31, a3[4], a3[5]);
    sub_223E0D0(v33, "\"", 1);
    if ( *((int *)a3 + 17) > 1 )
    {
      do
      {
        sub_223E0D0(qword_4FD4BE0, " ", 1);
        v37 = a3[9];
        v38 = 8LL * v32;
        if ( *(_QWORD *)(v37 + v38) )
        {
          src = *(const char **)(v37 + 8LL * v32);
          v34 = strlen(src);
          sub_223E0D0(qword_4FD4BE0, src, v34);
        }
        else
        {
          sub_222DC80(
            (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
            *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
        }
        if ( !strcmp(*(const char **)(a3[9] + 8LL * v32), "--orig_src_file_name")
          || !strcmp(*(const char **)(a3[9] + 8LL * v32), "--orig_src_path_name")
          || !strcmp(*(const char **)(a3[9] + 8LL * v32), "--compiler_bindir")
          || !strcmp(*(const char **)(a3[9] + 8LL * v32), "--sdk_dir") )
        {
          ++v32;
          sub_223E0D0(qword_4FD4BE0, " \"", 2);
          v35 = *(const char **)(a3[9] + v38 + 8);
          if ( v35 )
          {
            v36 = strlen(v35);
            sub_223E0D0(qword_4FD4BE0, v35, v36);
          }
          else
          {
            sub_222DC80(
              (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
              *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
          }
          sub_223E0D0(qword_4FD4BE0, "\"", 1);
        }
        ++v32;
      }
      while ( *((_DWORD *)a3 + 17) > v32 );
    }
    sub_223E0D0(qword_4FD4BE0, " ", 1);
    sub_223E0D0(qword_4FD4BE0, "-nvvm-version=nvvm70", 20);
    a2 = (size_t)"\n";
    sub_223E0D0(qword_4FD4BE0, "\n", 1);
  }
  v10 = *((_DWORD *)a3 + 17) + 1;
  v11 = 8LL * v10;
  if ( (unsigned __int64)v10 > 0xFFFFFFFFFFFFFFFLL )
    v11 = -1;
  v12 = sub_2207820(v11);
  v16 = *((int *)a3 + 17);
  v17 = v12;
  if ( (int)v16 > 0 )
  {
    a2 = a3[9];
    for ( i = 0; i != v16; ++i )
    {
      v13 = *(_QWORD *)(a2 + 8 * i);
      *(_QWORD *)(v17 + 8 * i) = v13;
    }
  }
  v19 = *a3;
  v20 = a3[1];
  *(_QWORD *)(v17 + 8 * v16) = *a3;
  if ( sub_16DA870(v11, a2, v13, v16, v14, v15) )
  {
    a2 = 18;
    v11 = (__int64)"CUDA C++ Front-End";
    sub_16DB3F0("CUDA C++ Front-End", 18, v19, v20);
  }
  if ( a8 )
  {
    if ( v17 )
    {
      v11 = v17;
      v25 = 0;
      j_j___libc_free_0_0(v17);
      goto LABEL_12;
    }
    goto LABEL_52;
  }
  v11 = (unsigned int)v10;
  a2 = v17;
  v39 = sub_12681D0((unsigned int)v10, v17, v47);
  v25 = (_QWORD *)v39;
  if ( v39 )
  {
    v23 = (_BYTE *)a3[4];
    if ( !v23 )
    {
      LOBYTE(v54[0]) = 0;
      v42 = *(_BYTE **)(v39 + 176);
      v21 = 0;
      v52 = v54;
      goto LABEL_42;
    }
    v40 = a3[5];
    v52 = v54;
    v51 = v40;
    if ( v40 > 0xF )
    {
      srca = v23;
      v45 = sub_22409D0(&v52, &v51, 0);
      v23 = srca;
      v52 = (_QWORD *)v45;
      v46 = (_QWORD *)v45;
      v54[0] = v51;
    }
    else
    {
      if ( v40 == 1 )
      {
        LOBYTE(v54[0]) = *v23;
        v41 = v54;
LABEL_37:
        n = v40;
        v23 = v25 + 24;
        *((_BYTE *)v41 + v40) = 0;
        v42 = (_BYTE *)v25[22];
        v21 = (size_t)v52;
        v43 = v42;
        if ( v52 != v54 )
        {
          v22 = v54[0];
          a2 = n;
          if ( v42 == v23 )
          {
            v25[22] = v52;
            v25[23] = a2;
            v25[24] = v22;
          }
          else
          {
            v44 = v25[24];
            v25[22] = v52;
            v25[23] = a2;
            v25[24] = v22;
            if ( v43 )
            {
              v52 = v43;
              v54[0] = v44;
              goto LABEL_43;
            }
          }
          v52 = v54;
          v43 = v54;
LABEL_43:
          n = 0;
          *v43 = 0;
          v11 = (__int64)v52;
          if ( v52 != v54 )
          {
            a2 = v54[0] + 1LL;
            j_j___libc_free_0(v52, v54[0] + 1LL);
          }
          goto LABEL_45;
        }
        v21 = n;
        if ( n )
        {
          if ( n == 1 )
          {
            *v42 = v54[0];
          }
          else
          {
            a2 = (size_t)v54;
            memcpy(v42, v54, n);
          }
          v21 = n;
          v42 = (_BYTE *)v25[22];
        }
LABEL_42:
        v25[23] = v21;
        v42[v21] = 0;
        v43 = v52;
        goto LABEL_43;
      }
      if ( !v40 )
      {
        v41 = v54;
        goto LABEL_37;
      }
      v46 = v54;
    }
    a2 = (size_t)v23;
    memcpy(v46, v23, v40);
    v40 = v51;
    v41 = v52;
    goto LABEL_37;
  }
LABEL_45:
  if ( v17 )
  {
    v11 = v17;
    j_j___libc_free_0_0(v17);
  }
  if ( *((_BYTE *)a3 + 66) )
  {
    v11 = a3[4];
    a2 = (size_t)v25;
    sub_1265320((char *)v11, (__int64)v25, 1u, v22, (__int64)v23);
  }
  if ( a9 )
  {
    if ( v25 )
    {
      sub_1633490(v25);
      a2 = 736;
      v11 = (__int64)v25;
      j_j___libc_free_0(v25, 736);
    }
LABEL_52:
    v25 = 0;
  }
LABEL_12:
  *a5 = 0;
  if ( sub_16DA870(v11, a2, v21, v22, v23, v24) )
    sub_16DB5E0();
  return v25;
}
