// Function: sub_2C72F80
// Address: 0x2c72f80
//
__int64 sub_2C72F80()
{
  _QWORD *v0; // r12
  _QWORD *v1; // rbx
  unsigned __int64 v2; // rsi
  _QWORD *v3; // rax
  _QWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  _QWORD *v11; // r12
  _QWORD *v12; // rbx
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rdx
  _QWORD *v22; // r12
  _QWORD *v23; // rbx
  unsigned __int64 v24; // rsi
  _QWORD *v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rax
  _QWORD *v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rdx
  _QWORD *v33; // r12
  _QWORD *v34; // rbx
  unsigned __int64 v35; // rsi
  _QWORD *v36; // rax
  _QWORD *v37; // rdi
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rax
  _QWORD *v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v45; // rdx
  char *v46; // rcx
  __int64 v47; // rdx
  char *v48; // rcx
  __int64 v49; // rdx
  char *v50; // rcx
  __int64 v51; // rdx
  char *v52; // rcx
  _QWORD *v53; // [rsp+0h] [rbp-30h] BYREF
  __int64 v54; // [rsp+8h] [rbp-28h]
  _BYTE v55[32]; // [rsp+10h] [rbp-20h] BYREF

  v0 = sub_C52410();
  v1 = v0 + 1;
  v2 = sub_C959E0();
  v3 = (_QWORD *)v0[2];
  if ( v3 )
  {
    v4 = v0 + 1;
    do
    {
      while ( 1 )
      {
        v5 = v3[2];
        v6 = v3[3];
        if ( v2 <= v3[4] )
          break;
        v3 = (_QWORD *)v3[3];
        if ( !v6 )
          goto LABEL_6;
      }
      v4 = v3;
      v3 = (_QWORD *)v3[2];
    }
    while ( v5 );
LABEL_6:
    if ( v1 != v4 && v2 >= v4[4] )
      v1 = v4;
  }
  if ( v1 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v7 = v1[7];
    if ( v7 )
    {
      v8 = v1 + 6;
      do
      {
        while ( 1 )
        {
          v9 = *(_QWORD *)(v7 + 16);
          v10 = *(_QWORD *)(v7 + 24);
          if ( *(_DWORD *)(v7 + 32) >= dword_5010F88 )
            break;
          v7 = *(_QWORD *)(v7 + 24);
          if ( !v10 )
            goto LABEL_15;
        }
        v8 = (_QWORD *)v7;
        v7 = *(_QWORD *)(v7 + 16);
      }
      while ( v9 );
LABEL_15:
      if ( v1 + 6 != v8 && dword_5010F88 >= *((_DWORD *)v8 + 8) )
      {
        if ( *((_DWORD *)v8 + 9) )
        {
          v54 = 0;
          v53 = v55;
          v55[0] = 0;
          sub_2241490((unsigned __int64 *)&v53, "option -nv-ocl is deprecated", 0x1Cu);
          sub_CEB590(&v53, 1, v49, v50);
          if ( v53 != (_QWORD *)v55 )
            j_j___libc_free_0((unsigned __int64)v53);
        }
      }
    }
  }
  v11 = sub_C52410();
  v12 = v11 + 1;
  v13 = sub_C959E0();
  v14 = (_QWORD *)v11[2];
  if ( v14 )
  {
    v15 = v11 + 1;
    do
    {
      while ( 1 )
      {
        v16 = v14[2];
        v17 = v14[3];
        if ( v13 <= v14[4] )
          break;
        v14 = (_QWORD *)v14[3];
        if ( !v17 )
          goto LABEL_22;
      }
      v15 = v14;
      v14 = (_QWORD *)v14[2];
    }
    while ( v16 );
LABEL_22:
    if ( v12 != v15 && v13 >= v15[4] )
      v12 = v15;
  }
  if ( v12 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v18 = v12[7];
    if ( v18 )
    {
      v19 = v12 + 6;
      do
      {
        while ( 1 )
        {
          v20 = *(_QWORD *)(v18 + 16);
          v21 = *(_QWORD *)(v18 + 24);
          if ( *(_DWORD *)(v18 + 32) >= dword_5010EA8 )
            break;
          v18 = *(_QWORD *)(v18 + 24);
          if ( !v21 )
            goto LABEL_31;
        }
        v19 = (_QWORD *)v18;
        v18 = *(_QWORD *)(v18 + 16);
      }
      while ( v20 );
LABEL_31:
      if ( v12 + 6 != v19 && dword_5010EA8 >= *((_DWORD *)v19 + 8) )
      {
        if ( *((_DWORD *)v19 + 9) )
        {
          v54 = 0;
          v53 = v55;
          v55[0] = 0;
          sub_2241490((unsigned __int64 *)&v53, "option -nv-cuda is deprecated", 0x1Du);
          sub_CEB590(&v53, 1, v51, v52);
          if ( v53 != (_QWORD *)v55 )
            j_j___libc_free_0((unsigned __int64)v53);
        }
      }
    }
  }
  v22 = sub_C52410();
  v23 = v22 + 1;
  v24 = sub_C959E0();
  v25 = (_QWORD *)v22[2];
  if ( v25 )
  {
    v26 = v22 + 1;
    do
    {
      while ( 1 )
      {
        v27 = v25[2];
        v28 = v25[3];
        if ( v24 <= v25[4] )
          break;
        v25 = (_QWORD *)v25[3];
        if ( !v28 )
          goto LABEL_38;
      }
      v26 = v25;
      v25 = (_QWORD *)v25[2];
    }
    while ( v27 );
LABEL_38:
    if ( v23 != v26 && v24 >= v26[4] )
      v23 = v26;
  }
  if ( v23 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v29 = v23[7];
    if ( v29 )
    {
      v30 = v23 + 6;
      do
      {
        while ( 1 )
        {
          v31 = *(_QWORD *)(v29 + 16);
          v32 = *(_QWORD *)(v29 + 24);
          if ( *(_DWORD *)(v29 + 32) >= dword_5010DC8 )
            break;
          v29 = *(_QWORD *)(v29 + 24);
          if ( !v32 )
            goto LABEL_47;
        }
        v30 = (_QWORD *)v29;
        v29 = *(_QWORD *)(v29 + 16);
      }
      while ( v31 );
LABEL_47:
      if ( v23 + 6 != v30 && dword_5010DC8 >= *((_DWORD *)v30 + 8) )
      {
        if ( *((_DWORD *)v30 + 9) )
        {
          v54 = 0;
          v53 = v55;
          v55[0] = 0;
          sub_2241490((unsigned __int64 *)&v53, "option -drvcuda is deprecated", 0x1Du);
          sub_CEB590(&v53, 1, v45, v46);
          if ( v53 != (_QWORD *)v55 )
            j_j___libc_free_0((unsigned __int64)v53);
        }
      }
    }
  }
  v33 = sub_C52410();
  v34 = v33 + 1;
  v35 = sub_C959E0();
  v36 = (_QWORD *)v33[2];
  if ( v36 )
  {
    v37 = v33 + 1;
    do
    {
      while ( 1 )
      {
        v38 = v36[2];
        v39 = v36[3];
        if ( v35 <= v36[4] )
          break;
        v36 = (_QWORD *)v36[3];
        if ( !v39 )
          goto LABEL_54;
      }
      v37 = v36;
      v36 = (_QWORD *)v36[2];
    }
    while ( v38 );
LABEL_54:
    if ( v34 != v37 && v35 >= v37[4] )
      v34 = v37;
  }
  if ( v34 == (_QWORD *)((char *)sub_C52410() + 8) )
    return 0;
  v40 = v34[7];
  if ( !v40 )
    return 0;
  v41 = v34 + 6;
  do
  {
    while ( 1 )
    {
      v42 = *(_QWORD *)(v40 + 16);
      v43 = *(_QWORD *)(v40 + 24);
      if ( *(_DWORD *)(v40 + 32) >= dword_5010CE8 )
        break;
      v40 = *(_QWORD *)(v40 + 24);
      if ( !v43 )
        goto LABEL_63;
    }
    v41 = (_QWORD *)v40;
    v40 = *(_QWORD *)(v40 + 16);
  }
  while ( v42 );
LABEL_63:
  if ( v34 + 6 == v41 )
    return 0;
  if ( dword_5010CE8 < *((_DWORD *)v41 + 8) )
    return 0;
  if ( !*((_DWORD *)v41 + 9) )
    return 0;
  v54 = 0;
  v53 = v55;
  v55[0] = 0;
  sub_2241490((unsigned __int64 *)&v53, "option -drvnvcl is deprecated", 0x1Du);
  sub_CEB590(&v53, 1, v47, v48);
  if ( v53 == (_QWORD *)v55 )
    return 0;
  j_j___libc_free_0((unsigned __int64)v53);
  return 0;
}
