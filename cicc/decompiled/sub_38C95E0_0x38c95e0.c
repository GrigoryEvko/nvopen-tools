// Function: sub_38C95E0
// Address: 0x38c95e0
//
__int64 __fastcall sub_38C95E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        _DWORD *a6,
        _DWORD *a7)
{
  _BYTE *v10; // rax
  unsigned __int64 v11; // rdx
  size_t v12; // r13
  unsigned __int64 v13; // r14
  char v14; // si
  char v15; // al
  char *v16; // rax
  _BYTE *v17; // rax
  int v18; // r8d
  __int64 v19; // rax
  unsigned int v20; // r12d
  _BYTE *v21; // rax
  _BYTE *v22; // rax
  _BYTE *v23; // rax
  unsigned __int64 v24; // rcx
  _BYTE *v26; // rax
  __int64 v27; // r14
  char v28; // r13
  char v29; // si
  char *v30; // rax
  char v31; // al
  _BYTE *v32; // rax
  unsigned __int8 v34; // [rsp+2Fh] [rbp-51h] BYREF
  char *v35[10]; // [rsp+30h] [rbp-50h] BYREF

  if ( a3 != 0x7FFFFFFFFFFFFFFFLL )
  {
    v26 = *(_BYTE **)(a5 + 24);
    if ( *(_QWORD *)(a5 + 16) <= (unsigned __int64)v26 )
    {
      sub_16E7DE0(a5, 3);
    }
    else
    {
      *(_QWORD *)(a5 + 24) = v26 + 1;
      *v26 = 3;
    }
    v27 = a3;
    while ( 1 )
    {
      v31 = v27;
      v29 = v27 & 0x7F;
      v27 >>= 7;
      if ( !v27 )
        break;
      if ( v27 != -1 )
        goto LABEL_29;
      v28 = 0;
      if ( (v31 & 0x40) == 0 )
        goto LABEL_29;
      v30 = *(char **)(a5 + 24);
      if ( (unsigned __int64)v30 >= *(_QWORD *)(a5 + 16) )
      {
LABEL_37:
        sub_16E7DE0(a5, v29);
        goto LABEL_32;
      }
LABEL_31:
      *(_QWORD *)(a5 + 24) = v30 + 1;
      *v30 = v29;
LABEL_32:
      if ( !v28 )
        goto LABEL_2;
    }
    v28 = 0;
    if ( (v31 & 0x40) != 0 )
    {
LABEL_29:
      v29 |= 0x80u;
      v28 = 1;
    }
    v30 = *(char **)(a5 + 24);
    if ( (unsigned __int64)v30 >= *(_QWORD *)(a5 + 16) )
      goto LABEL_37;
    goto LABEL_31;
  }
LABEL_2:
  v10 = *(_BYTE **)(a5 + 24);
  v11 = *(_QWORD *)(a5 + 16);
  if ( a4 <= 0xEA60 )
  {
    if ( v11 <= (unsigned __int64)v10 )
    {
      sub_16E7DE0(a5, 9);
    }
    else
    {
      *(_QWORD *)(a5 + 24) = v10 + 1;
      *v10 = 9;
    }
    *a6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a5 + 64LL))(a5) + *(_DWORD *)(a5 + 24) - *(_DWORD *)(a5 + 8);
    *a7 = 2;
    v21 = *(_BYTE **)(a5 + 24);
    if ( (unsigned __int64)v21 >= *(_QWORD *)(a5 + 16) )
    {
      sub_16E7DE0(a5, 0);
      v22 = *(_BYTE **)(a5 + 24);
      if ( (unsigned __int64)v22 < *(_QWORD *)(a5 + 16) )
        goto LABEL_20;
    }
    else
    {
      *(_QWORD *)(a5 + 24) = v21 + 1;
      *v21 = 0;
      v22 = *(_BYTE **)(a5 + 24);
      if ( (unsigned __int64)v22 < *(_QWORD *)(a5 + 16) )
      {
LABEL_20:
        v20 = 1;
        *(_QWORD *)(a5 + 24) = v22 + 1;
        *v22 = 0;
        goto LABEL_21;
      }
    }
    v20 = 1;
    sub_16E7DE0(a5, 0);
    goto LABEL_21;
  }
  v12 = *(unsigned int *)(*(_QWORD *)(a1 + 16) + 8LL);
  if ( v11 <= (unsigned __int64)v10 )
  {
    sub_16E7DE0(a5, 0);
  }
  else
  {
    *(_QWORD *)(a5 + 24) = v10 + 1;
    *v10 = 0;
  }
  v13 = (unsigned int)(v12 + 1);
  do
  {
    while ( 1 )
    {
      v14 = v13 & 0x7F;
      v15 = v13 & 0x7F | 0x80;
      v13 >>= 7;
      if ( v13 )
        v14 = v15;
      v16 = *(char **)(a5 + 24);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(a5 + 16) )
        break;
      *(_QWORD *)(a5 + 24) = v16 + 1;
      *v16 = v14;
      if ( !v13 )
        goto LABEL_11;
    }
    sub_16E7DE0(a5, v14);
  }
  while ( v13 );
LABEL_11:
  v17 = *(_BYTE **)(a5 + 24);
  if ( (unsigned __int64)v17 >= *(_QWORD *)(a5 + 16) )
  {
    sub_16E7DE0(a5, 2);
  }
  else
  {
    *(_QWORD *)(a5 + 24) = v17 + 1;
    *v17 = 2;
  }
  v18 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a5 + 64LL))(a5);
  v19 = *(_QWORD *)(a5 + 24) - *(_QWORD *)(a5 + 8);
  v35[0] = 0;
  *a6 = v18 + v19;
  v35[1] = 0;
  *a7 = v12;
  v35[2] = 0;
  v34 = 0;
  sub_CD1880((__int64 *)v35, 0, v12, &v34);
  sub_16E7EE0(a5, v35[0], v12);
  if ( v35[0] )
    j_j___libc_free_0((unsigned __int64)v35[0]);
  v20 = 0;
LABEL_21:
  v23 = *(_BYTE **)(a5 + 24);
  v24 = *(_QWORD *)(a5 + 16);
  if ( a3 == 0x7FFFFFFFFFFFFFFFLL )
  {
    if ( (unsigned __int64)v23 >= v24 )
    {
      sub_16E7DE0(a5, 0);
    }
    else
    {
      *(_QWORD *)(a5 + 24) = v23 + 1;
      *v23 = 0;
    }
    v32 = *(_BYTE **)(a5 + 24);
    if ( (unsigned __int64)v32 >= *(_QWORD *)(a5 + 16) )
    {
      sub_16E7DE0(a5, 1);
    }
    else
    {
      *(_QWORD *)(a5 + 24) = v32 + 1;
      *v32 = 1;
    }
    v23 = *(_BYTE **)(a5 + 24);
    if ( (unsigned __int64)v23 < *(_QWORD *)(a5 + 16) )
      goto LABEL_23;
LABEL_43:
    sub_16E7DE0(a5, 1);
    return v20;
  }
  if ( (unsigned __int64)v23 >= v24 )
    goto LABEL_43;
LABEL_23:
  *(_QWORD *)(a5 + 24) = v23 + 1;
  *v23 = 1;
  return v20;
}
