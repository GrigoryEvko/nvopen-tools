// Function: sub_721C20
// Address: 0x721c20
//
char __fastcall sub_721C20(_QWORD *a1, char *a2)
{
  char *v2; // r12
  __int64 v3; // rax
  char v4; // bl
  _DWORD *v5; // r8
  bool v6; // r9
  char *v7; // r13
  int v8; // r14d
  char *v9; // r12
  unsigned __int64 v10; // r9
  char v11; // r14
  char *v12; // rbx
  __int64 v13; // rax
  _DWORD *v15; // [rsp+0h] [rbp-40h]
  _DWORD *v16; // [rsp+0h] [rbp-40h]
  _DWORD *v17; // [rsp+0h] [rbp-40h]
  unsigned __int64 v18; // [rsp+8h] [rbp-38h]
  _DWORD *v19; // [rsp+8h] [rbp-38h]
  bool v20; // [rsp+8h] [rbp-38h]
  _DWORD *v21; // [rsp+8h] [rbp-38h]

  v2 = a2;
  v3 = (__int64)&dword_4F07594;
  v4 = *a2;
  if ( !dword_4F07594 )
    goto LABEL_5;
  v5 = &dword_4F07598;
  LODWORD(v3) = dword_4F07598;
  if ( dword_4F07598 )
  {
    if ( v4 != 92 && v4 != 47 )
      goto LABEL_5;
    LOBYTE(v3) = a2[1];
    if ( (_BYTE)v3 != 92 && (_BYTE)v3 != 47 )
      goto LABEL_5;
    goto LABEL_59;
  }
  if ( v4 != 47 )
    goto LABEL_5;
  if ( a2[1] == 47 )
  {
LABEL_59:
    v3 = a1[2];
    if ( (unsigned __int64)(v3 + 1) > a1[1] )
    {
      sub_823810(a1);
      v3 = a1[2];
    }
    *(_BYTE *)(a1[4] + v3) = 47;
    ++a1[2];
    v4 = *a2;
LABEL_5:
    if ( v4 )
    {
      v5 = &dword_4F07598;
      LODWORD(v3) = dword_4F07598;
      goto LABEL_7;
    }
    return v3;
  }
LABEL_7:
  while ( 2 )
  {
    LOBYTE(v3) = (_DWORD)v3 != 0;
    v6 = v3 & (v4 == 92);
    if ( !v6 )
      v6 = v4 == 47;
    while ( v4 == 92 && (_BYTE)v3 || v4 == 47 )
      v4 = *++v2;
    if ( !v4 )
      return v3;
    v7 = v2;
    do
    {
      v3 = 1;
      if ( v4 < 0 )
      {
        v17 = v5;
        v20 = v6;
        LODWORD(v3) = sub_721AB0(v7, 0, 0);
        v5 = v17;
        v6 = v20;
        v3 = (int)v3;
      }
      v7 += v3;
      v4 = *v7;
      if ( !*v7 )
        goto LABEL_20;
      if ( *v5 && v4 == 92 )
      {
        v4 = 92;
LABEL_20:
        v8 = (_DWORD)v7 - (_DWORD)v2;
        if ( (_DWORD)v7 - (_DWORD)v2 != 1 )
          goto LABEL_21;
LABEL_47:
        if ( *v2 == 46 )
          goto LABEL_42;
LABEL_48:
        v13 = a1[2];
        if ( v6 || v13 )
        {
          if ( (unsigned __int64)(v13 + 1) > a1[1] )
          {
            v21 = v5;
            sub_823810(a1);
            v13 = a1[2];
            v5 = v21;
          }
          *(_BYTE *)(a1[4] + v13) = 47;
          ++a1[2];
        }
        v19 = v5;
        LOBYTE(v3) = sub_8238B0(a1, v2, (unsigned int)v8);
        v4 = *v7;
        v5 = v19;
        if ( *v7 )
          goto LABEL_43;
        return v3;
      }
    }
    while ( v4 != 47 );
    v8 = (_DWORD)v7 - (_DWORD)v2;
    if ( (_DWORD)v7 - (_DWORD)v2 == 1 )
      goto LABEL_47;
LABEL_21:
    if ( v8 != 2 )
    {
      if ( v8 <= 0 )
        goto LABEL_42;
      goto LABEL_48;
    }
    if ( *v2 != 46 || v2[1] != 46 )
      goto LABEL_48;
    v3 = a1[2];
    if ( !v3 )
      goto LABEL_42;
    v9 = (char *)a1[4];
    v10 = (unsigned __int64)&v9[v3 - 1];
    if ( dword_4F07594 && v3 == 2 )
    {
      v15 = v5;
      v11 = *v9;
      LODWORD(v3) = isalpha((unsigned __int8)*v9);
      v10 = (unsigned __int64)(v9 + 1);
      v5 = v15;
      if ( (_DWORD)v3 && v9[1] == 58 || v11 == 92 && v9[1] == 92 )
        goto LABEL_42;
      if ( v9 >= v9 + 1 )
        goto LABEL_69;
    }
    else
    {
      if ( (unsigned __int64)v9 >= v10 )
        goto LABEL_69;
      v11 = *v9;
    }
    v12 = 0;
    while ( 2 )
    {
      if ( *v5 && v11 == 92 || v11 == 47 )
      {
        LOBYTE(v3) = 1;
        v12 = v9++;
        if ( v10 <= (unsigned __int64)v9 )
          break;
        goto LABEL_35;
      }
      v3 = 1;
      if ( v11 < 0 )
      {
        v16 = v5;
        v18 = v10;
        LODWORD(v3) = sub_721AB0(v9, 0, 0);
        v10 = v18;
        v5 = v16;
        v3 = (int)v3;
      }
      v9 += v3;
      if ( v10 > (unsigned __int64)v9 )
      {
LABEL_35:
        v11 = *v9;
        continue;
      }
      break;
    }
    if ( v12 )
    {
      a1[2] += ~(v10 - (_QWORD)v12);
      v4 = *v7;
      goto LABEL_42;
    }
LABEL_69:
    a1[2] = 0;
    v4 = *v7;
LABEL_42:
    if ( v4 )
    {
LABEL_43:
      LODWORD(v3) = *v5;
      v2 = v7;
      continue;
    }
    return v3;
  }
}
