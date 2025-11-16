// Function: sub_905880
// Address: 0x905880
//
_QWORD *__fastcall sub_905880(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 a4,
        _DWORD *a5,
        char a6,
        int a7,
        char a8,
        char a9)
{
  int v10; // r13d
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r14
  __int64 v15; // rsi
  __int64 i; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r12
  _QWORD *v20; // r13
  const char **v22; // rax
  const char *v23; // r13
  size_t v24; // rax
  __int64 v25; // r13
  __int64 v26; // rdi
  int v27; // r13d
  __int64 v28; // rax
  size_t v29; // rax
  const char *v30; // r14
  size_t v31; // rax
  __int64 v32; // rax
  __int64 v33; // r14
  _BYTE *v34; // r8
  size_t v35; // r15
  _QWORD *v36; // rax
  _BYTE *v37; // rdi
  size_t v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // r8
  __int64 v41; // rdx
  __int64 v42; // rax
  _QWORD *v43; // rdi
  size_t v44; // rdx
  const char *src; // [rsp+28h] [rbp-68h]
  _BYTE *srca; // [rsp+28h] [rbp-68h]
  size_t v49; // [rsp+38h] [rbp-58h] BYREF
  _QWORD *v50; // [rsp+40h] [rbp-50h] BYREF
  size_t n; // [rsp+48h] [rbp-48h]
  _QWORD v52[8]; // [rsp+50h] [rbp-40h] BYREF

  if ( a6 )
  {
    sub_223E0D0(qword_4FD4BE0, "\"", 1);
    v22 = (const char **)a3[9];
    v23 = *v22;
    if ( *v22 )
    {
      v24 = strlen(*v22);
      sub_223E0D0(qword_4FD4BE0, v23, v24);
    }
    else
    {
      sub_222DC80(
        (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
        *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
    }
    sub_223E0D0(qword_4FD4BE0, "\" -lgenfe \"", 11);
    v25 = sub_223E0D0(qword_4FD4BE0, *a3, a3[1]);
    sub_223E0D0(v25, "\" -o \"", 6);
    v26 = v25;
    v27 = 1;
    v28 = sub_223E0D0(v26, a3[4], a3[5]);
    sub_223E0D0(v28, "\"", 1);
    if ( *((int *)a3 + 17) > 1 )
    {
      do
      {
        sub_223E0D0(qword_4FD4BE0, " ", 1);
        v32 = a3[9];
        v33 = 8LL * v27;
        if ( *(_QWORD *)(v32 + v33) )
        {
          src = *(const char **)(v32 + 8LL * v27);
          v29 = strlen(src);
          sub_223E0D0(qword_4FD4BE0, src, v29);
        }
        else
        {
          sub_222DC80(
            (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
            *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
        }
        if ( !strcmp(*(const char **)(a3[9] + 8LL * v27), "--orig_src_file_name")
          || !strcmp(*(const char **)(a3[9] + 8LL * v27), "--orig_src_path_name")
          || !strcmp(*(const char **)(a3[9] + 8LL * v27), "--compiler_bindir")
          || !strcmp(*(const char **)(a3[9] + 8LL * v27), "--sdk_dir") )
        {
          ++v27;
          sub_223E0D0(qword_4FD4BE0, " \"", 2);
          v30 = *(const char **)(a3[9] + v33 + 8);
          if ( v30 )
          {
            v31 = strlen(v30);
            sub_223E0D0(qword_4FD4BE0, v30, v31);
          }
          else
          {
            sub_222DC80(
              (char *)qword_4FD4BE0 + *(_QWORD *)(qword_4FD4BE0[0] - 24LL),
              *(_DWORD *)((char *)&qword_4FD4BE0[4] + *(_QWORD *)(qword_4FD4BE0[0] - 24LL)) | 1u);
          }
          sub_223E0D0(qword_4FD4BE0, "\"", 1);
        }
        ++v27;
      }
      while ( *((_DWORD *)a3 + 17) > v27 );
    }
    sub_223E0D0(qword_4FD4BE0, " ", 1);
    sub_223E0D0(qword_4FD4BE0, "-nvvm-version=nvvm-latest", 25);
    sub_223E0D0(qword_4FD4BE0, "\n", 1);
  }
  v10 = *((_DWORD *)a3 + 17) + 1;
  v11 = 8LL * v10;
  if ( (unsigned __int64)v10 > 0xFFFFFFFFFFFFFFFLL )
    v11 = -1;
  v12 = sub_2207820(v11);
  v13 = *((int *)a3 + 17);
  v14 = v12;
  if ( (int)v13 <= 0 )
  {
    v41 = *a3;
    *(_QWORD *)(v12 + 8 * v13) = *a3;
    v19 = sub_C996C0("CUDA C++ Front-End", 18, v41, a3[1]);
    if ( a8 )
    {
      if ( v14 )
      {
LABEL_8:
        v20 = 0;
        j_j___libc_free_0_0(v14);
        goto LABEL_9;
      }
      goto LABEL_48;
    }
  }
  else
  {
    v15 = a3[9];
    for ( i = 0; i != v13; ++i )
      *(_QWORD *)(v14 + 8 * i) = *(_QWORD *)(v15 + 8 * i);
    v17 = *a3;
    v18 = a3[1];
    *(_QWORD *)(v14 + 8 * i) = *a3;
    v19 = sub_C996C0("CUDA C++ Front-End", 18, v17, v18);
    if ( a8 )
      goto LABEL_8;
  }
  v20 = (_QWORD *)sub_908750((unsigned int)v10, v14, a2);
  if ( v20 )
  {
    v34 = (_BYTE *)a3[4];
    v35 = a3[5];
    v50 = v52;
    if ( &v34[v35] && !v34 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v49 = v35;
    if ( v35 > 0xF )
    {
      srca = v34;
      v42 = sub_22409D0(&v50, &v49, 0);
      v34 = srca;
      v50 = (_QWORD *)v42;
      v43 = (_QWORD *)v42;
      v52[0] = v49;
    }
    else
    {
      if ( v35 == 1 )
      {
        LOBYTE(v52[0]) = *v34;
        v36 = v52;
        goto LABEL_35;
      }
      if ( !v35 )
      {
        v36 = v52;
        goto LABEL_35;
      }
      v43 = v52;
    }
    memcpy(v43, v34, v35);
    v35 = v49;
    v36 = v50;
LABEL_35:
    n = v35;
    *((_BYTE *)v36 + v35) = 0;
    v37 = (_BYTE *)v20[21];
    if ( v50 == v52 )
    {
      v44 = n;
      if ( n )
      {
        if ( n == 1 )
          *v37 = v52[0];
        else
          memcpy(v37, v52, n);
        v44 = n;
        v37 = (_BYTE *)v20[21];
      }
      v20[22] = v44;
      v37[v44] = 0;
      v37 = v50;
      goto LABEL_39;
    }
    v38 = n;
    v39 = v52[0];
    if ( v37 == (_BYTE *)(v20 + 23) )
    {
      v20[21] = v50;
      v20[22] = v38;
      v20[23] = v39;
    }
    else
    {
      v40 = v20[23];
      v20[21] = v50;
      v20[22] = v38;
      v20[23] = v39;
      if ( v37 )
      {
        v50 = v37;
        v52[0] = v40;
        goto LABEL_39;
      }
    }
    v50 = v52;
    v37 = v52;
LABEL_39:
    n = 0;
    *v37 = 0;
    if ( v50 != v52 )
      j_j___libc_free_0(v50, v52[0] + 1LL);
  }
  if ( v14 )
    j_j___libc_free_0_0(v14);
  if ( *((_BYTE *)a3 + 66) )
    sub_905860((const char *)a3[4], (__int64)v20, 1);
  if ( a9 )
  {
    if ( v20 )
    {
      sub_BA9C10(v20);
      j_j___libc_free_0(v20, 880);
    }
LABEL_48:
    v20 = 0;
  }
LABEL_9:
  *a5 = 0;
  if ( v19 )
    sub_C9AF60(v19);
  return v20;
}
