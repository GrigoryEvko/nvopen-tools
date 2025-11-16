// Function: sub_C2FB70
// Address: 0xc2fb70
//
__int64 __fastcall sub_C2FB70(_BYTE *a1, __int64 a2)
{
  char v3; // al
  __int64 v4; // r12
  _BYTE *v5; // rbx
  unsigned __int64 v6; // rax
  __int64 v7; // rdi
  unsigned __int64 v8; // rsi
  char v9; // al
  _BYTE *v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rdx
  __int64 v15; // rdx
  char v16; // al
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  char *v19; // rax
  unsigned __int64 v20; // rdx
  _BYTE *v21; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v22; // [rsp+8h] [rbp-28h]

  if ( !a2 )
    goto LABEL_7;
  if ( a2 == 1 )
  {
    v3 = *a1;
    if ( *a1 == 43 || v3 == 45 )
      goto LABEL_7;
LABEL_10:
    v4 = a2;
    v5 = a1;
    if ( ((v3 - 43) & 0xFD) != 0 )
      goto LABEL_12;
    goto LABEL_11;
  }
  if ( a2 != 4 )
  {
    v3 = *a1;
    goto LABEL_10;
  }
  if ( *(_DWORD *)a1 == 1851878958 || *(_DWORD *)a1 == 1314999854 || *(_DWORD *)a1 == 1312902702 )
    return 1;
  if ( ((*a1 - 43) & 0xFD) == 0 )
  {
LABEL_11:
    v4 = a2 - 1;
    v5 = a1 + 1;
LABEL_12:
    if ( v4 != 4 )
      goto LABEL_13;
    goto LABEL_26;
  }
  v5 = a1;
LABEL_26:
  if ( *(_DWORD *)v5 == 1718511918 )
    return 1;
  if ( *(_DWORD *)v5 == 1718503726 )
    return 1;
  v4 = 4;
  if ( *(_DWORD *)v5 == 1179535662 )
    return 1;
LABEL_13:
  if ( a2 != 1 )
  {
    if ( *(_WORD *)a1 == 28464 )
    {
      if ( a2 != 2 )
      {
        v21 = a1 + 2;
        v22 = a2 - 2;
        LOBYTE(v4) = sub_C935B0(&v21, "01234567", 8, 0) == -1;
        return (unsigned int)v4;
      }
      goto LABEL_7;
    }
    if ( *(_WORD *)a1 == 30768 )
    {
      if ( a2 != 2 )
      {
        v21 = a1 + 2;
        v22 = a2 - 2;
        LOBYTE(v4) = sub_C935B0(&v21, "0123456789abcdefABCDEF", 22, 0) == -1;
        return (unsigned int)v4;
      }
      goto LABEL_7;
    }
  }
  if ( v4 )
  {
    v16 = *v5;
    if ( *v5 == 46 )
    {
      if ( v4 == 1 || !strchr("0123456789", (char)v5[1]) )
        goto LABEL_7;
    }
    else if ( v16 == 69 || v16 == 101 )
    {
      goto LABEL_7;
    }
  }
  v22 = v4;
  v21 = v5;
  LODWORD(v4) = 1;
  v6 = sub_C935B0(&v21, "0123456789", 10, 0);
  if ( v6 >= v22 )
    return (unsigned int)v4;
  v7 = (__int64)&v21[v6];
  v8 = v22 - v6;
  v9 = v21[v6];
  if ( v9 == 46 )
  {
    v17 = sub_C2F470(v7, v8, 1u);
    v22 = v18;
    v21 = (_BYTE *)v17;
    v19 = (char *)sub_C2FB30(&v21, (__int64)"0123456789", 10);
    v7 = (__int64)v19;
    v8 = v20;
    if ( !v20 )
      return (unsigned int)v4;
    v9 = *v19;
  }
  if ( (v9 & 0xDF) == 0x45 )
  {
    v10 = (_BYTE *)sub_C2F470(v7, v8, 1u);
    v12 = (__int64)v10;
    v13 = v11;
    if ( v11 )
    {
      if ( ((*v10 - 43) & 0xFD) != 0 || (v12 = sub_C2F470((__int64)v10, v11, 1u), (v13 = v14) != 0) )
      {
        v21 = (_BYTE *)v12;
        v22 = v13;
        sub_C2FB30(&v21, (__int64)"0123456789", 10);
        LOBYTE(v4) = v15 == 0;
        return (unsigned int)v4;
      }
    }
  }
LABEL_7:
  LODWORD(v4) = 0;
  return (unsigned int)v4;
}
