// Function: sub_1BF83F0
// Address: 0x1bf83f0
//
__int64 sub_1BF83F0()
{
  unsigned __int64 v0; // rsi
  _QWORD *v1; // rax
  _DWORD *v2; // rdi
  __int64 v3; // rcx
  __int64 v4; // rdx
  unsigned __int64 v5; // rsi
  _QWORD *v6; // rax
  _DWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 v10; // rsi
  _QWORD *v11; // rax
  _DWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  _QWORD *v16; // rax
  _DWORD *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  _DWORD *v21; // r8
  _DWORD *v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v26; // rax
  _DWORD *v27; // r8
  _DWORD *v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  _DWORD *v32; // r8
  _DWORD *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rax
  _DWORD *v37; // r8
  _DWORD *v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rdx
  _QWORD *v41; // [rsp+0h] [rbp-40h] BYREF
  __int64 v42; // [rsp+8h] [rbp-38h]
  _QWORD v43[6]; // [rsp+10h] [rbp-30h] BYREF

  v0 = sub_16D5D50();
  v1 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v2 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v3 = v1[2];
        v4 = v1[3];
        if ( v0 <= v1[4] )
          break;
        v1 = (_QWORD *)v1[3];
        if ( !v4 )
          goto LABEL_6;
      }
      v2 = v1;
      v1 = (_QWORD *)v1[2];
    }
    while ( v3 );
LABEL_6:
    if ( v2 != dword_4FA0208 && v0 >= *((_QWORD *)v2 + 4) )
    {
      v26 = *((_QWORD *)v2 + 7);
      v27 = v2 + 12;
      if ( v26 )
      {
        v28 = v2 + 12;
        do
        {
          while ( 1 )
          {
            v29 = *(_QWORD *)(v26 + 16);
            v30 = *(_QWORD *)(v26 + 24);
            if ( *(_DWORD *)(v26 + 32) >= dword_4FB9D68 )
              break;
            v26 = *(_QWORD *)(v26 + 24);
            if ( !v30 )
              goto LABEL_42;
          }
          v28 = (_DWORD *)v26;
          v26 = *(_QWORD *)(v26 + 16);
        }
        while ( v29 );
LABEL_42:
        if ( v27 != v28 && dword_4FB9D68 >= v28[8] )
        {
          if ( v28[9] )
          {
            v42 = 0;
            v41 = v43;
            LOBYTE(v43[0]) = 0;
            sub_2241490(&v41, "option -nv-ocl is deprecated", 28, v29);
            sub_1C3EFD0(&v41, 1);
            if ( v41 != v43 )
              j_j___libc_free_0(v41, v43[0] + 1LL);
          }
        }
      }
    }
  }
  v5 = sub_16D5D50();
  v6 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v7 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v8 = v6[2];
        v9 = v6[3];
        if ( v5 <= v6[4] )
          break;
        v6 = (_QWORD *)v6[3];
        if ( !v9 )
          goto LABEL_13;
      }
      v7 = v6;
      v6 = (_QWORD *)v6[2];
    }
    while ( v8 );
LABEL_13:
    if ( v7 != dword_4FA0208 && v5 >= *((_QWORD *)v7 + 4) )
    {
      v36 = *((_QWORD *)v7 + 7);
      v37 = v7 + 12;
      if ( v36 )
      {
        v38 = v7 + 12;
        do
        {
          while ( 1 )
          {
            v39 = *(_QWORD *)(v36 + 16);
            v40 = *(_QWORD *)(v36 + 24);
            if ( *(_DWORD *)(v36 + 32) >= dword_4FB9C88 )
              break;
            v36 = *(_QWORD *)(v36 + 24);
            if ( !v40 )
              goto LABEL_62;
          }
          v38 = (_DWORD *)v36;
          v36 = *(_QWORD *)(v36 + 16);
        }
        while ( v39 );
LABEL_62:
        if ( v37 != v38 && dword_4FB9C88 >= v38[8] )
        {
          if ( v38[9] )
          {
            v42 = 0;
            v41 = v43;
            LOBYTE(v43[0]) = 0;
            sub_2241490(&v41, "option -nv-cuda is deprecated", 29, v39);
            sub_1C3EFD0(&v41, 1);
            if ( v41 != v43 )
              j_j___libc_free_0(v41, v43[0] + 1LL);
          }
        }
      }
    }
  }
  v10 = sub_16D5D50();
  v11 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v12 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v13 = v11[2];
        v14 = v11[3];
        if ( v10 <= v11[4] )
          break;
        v11 = (_QWORD *)v11[3];
        if ( !v14 )
          goto LABEL_20;
      }
      v12 = v11;
      v11 = (_QWORD *)v11[2];
    }
    while ( v13 );
LABEL_20:
    if ( v12 != dword_4FA0208 && v10 >= *((_QWORD *)v12 + 4) )
    {
      v31 = *((_QWORD *)v12 + 7);
      v32 = v12 + 12;
      if ( v31 )
      {
        v33 = v12 + 12;
        do
        {
          while ( 1 )
          {
            v34 = *(_QWORD *)(v31 + 16);
            v35 = *(_QWORD *)(v31 + 24);
            if ( *(_DWORD *)(v31 + 32) >= dword_4FB9BA8 )
              break;
            v31 = *(_QWORD *)(v31 + 24);
            if ( !v35 )
              goto LABEL_52;
          }
          v33 = (_DWORD *)v31;
          v31 = *(_QWORD *)(v31 + 16);
        }
        while ( v34 );
LABEL_52:
        if ( v32 != v33 && dword_4FB9BA8 >= v33[8] )
        {
          if ( v33[9] )
          {
            v42 = 0;
            v41 = v43;
            LOBYTE(v43[0]) = 0;
            sub_2241490(&v41, "option -drvcuda is deprecated", 29);
            sub_1C3EFD0(&v41, 1);
            if ( v41 != v43 )
              j_j___libc_free_0(v41, v43[0] + 1LL);
          }
        }
      }
    }
  }
  v15 = sub_16D5D50();
  v16 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    return 0;
  v17 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v18 = v16[2];
      v19 = v16[3];
      if ( v15 <= v16[4] )
        break;
      v16 = (_QWORD *)v16[3];
      if ( !v19 )
        goto LABEL_27;
    }
    v17 = v16;
    v16 = (_QWORD *)v16[2];
  }
  while ( v18 );
LABEL_27:
  if ( v17 == dword_4FA0208 )
    return 0;
  if ( v15 < *((_QWORD *)v17 + 4) )
    return 0;
  v20 = *((_QWORD *)v17 + 7);
  v21 = v17 + 12;
  if ( !v20 )
    return 0;
  v22 = v17 + 12;
  do
  {
    while ( 1 )
    {
      v23 = *(_QWORD *)(v20 + 16);
      v24 = *(_QWORD *)(v20 + 24);
      if ( *(_DWORD *)(v20 + 32) >= dword_4FB9AC8 )
        break;
      v20 = *(_QWORD *)(v20 + 24);
      if ( !v24 )
        goto LABEL_34;
    }
    v22 = (_DWORD *)v20;
    v20 = *(_QWORD *)(v20 + 16);
  }
  while ( v23 );
LABEL_34:
  if ( v21 == v22 )
    return 0;
  if ( dword_4FB9AC8 < v22[8] )
    return 0;
  if ( !v22[9] )
    return 0;
  v42 = 0;
  v41 = v43;
  LOBYTE(v43[0]) = 0;
  sub_2241490(&v41, "option -drvnvcl is deprecated", 29, v23);
  sub_1C3EFD0(&v41, 1);
  if ( v41 == v43 )
    return 0;
  j_j___libc_free_0(v41, v43[0] + 1LL);
  return 0;
}
