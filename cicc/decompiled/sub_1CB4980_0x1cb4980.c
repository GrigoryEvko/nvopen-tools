// Function: sub_1CB4980
// Address: 0x1cb4980
//
char __fastcall sub_1CB4980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  size_t v7; // rdx
  char **v9; // rax
  char *v10; // rsi
  __int64 v11; // r15
  size_t v12; // rdx
  __int64 v13; // rcx
  size_t v14; // rdx
  char *v15; // rbx
  unsigned __int8 v16; // r12
  size_t v17; // rdx
  char *v18; // r15
  size_t v19; // rax
  __int64 v21; // [rsp+8h] [rbp-48h]
  char *v23; // [rsp+18h] [rbp-38h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v7 = 0;
  v21 = *(_QWORD *)(a4 + 56);
  v9 = off_4CD4978;
  v10 = off_4CD4978[0];
  if ( off_4CD4978[0] )
  {
    v23 = off_4CD4978[0];
    v9 = (char **)strlen(off_4CD4978[0]);
    v10 = v23;
    v7 = (size_t)v9;
  }
  v11 = *(_QWORD *)(a2 + 48);
  if ( v11 || *(__int16 *)(a2 + 18) < 0 )
  {
    v11 = sub_1625940(a2, v10, v7);
    LOBYTE(v9) = *(_BYTE *)(a3 + 16);
    if ( (unsigned __int8)v9 <= 0x17u )
    {
      if ( v11 )
      {
        if ( (_BYTE)v9 == 17 )
        {
          v9 = *(char ***)a3;
          if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 15 )
          {
            v9 = *(char ***)(a3 + 8);
            if ( v9 )
            {
              if ( !v9[1] )
                LOBYTE(v9) = sub_15E0E40(a3, 20);
            }
          }
        }
      }
      return (char)v9;
    }
    v9 = off_4CD4978;
    v10 = off_4CD4978[0];
  }
  else if ( *(_BYTE *)(a3 + 16) <= 0x17u )
  {
    return (char)v9;
  }
  v12 = 0;
  if ( v10 )
  {
    v9 = (char **)strlen(v10);
    v12 = (size_t)v9;
  }
  if ( !*(_QWORD *)(a3 + 48) && *(__int16 *)(a3 + 18) >= 0 )
  {
    if ( !v11 )
      return (char)v9;
LABEL_15:
    if ( (unsigned __int8)(*(_BYTE *)(a3 + 16) - 71) > 1u || (v9 = *(char ***)(a3 + 8)) != 0 && !v9[1] )
    {
      v14 = 0;
      v15 = off_4CD4978[0];
      if ( off_4CD4978[0] )
        v14 = strlen(off_4CD4978[0]);
      sub_1626100(a3, v15, v14, v11);
      sub_1CCABF0(2, v21);
      LOBYTE(v9) = sub_1CB4290(a1, a3, a4, a5);
      while ( 1 )
      {
        a3 = *(_QWORD *)(a3 + 8);
        if ( !a3 )
          break;
        while ( 1 )
        {
          v9 = (char **)sub_1648700(a3);
          if ( *((_BYTE *)v9 + 16) <= 0x17u )
            break;
          LOBYTE(v9) = sub_1CB4290(a1, (__int64)v9, a4, a5);
          a3 = *(_QWORD *)(a3 + 8);
          if ( !a3 )
            return (char)v9;
        }
      }
    }
    return (char)v9;
  }
  v13 = sub_1625940(a3, v10, v12);
  LOBYTE(v9) = v13 == 0;
  if ( (v11 != 0) != (v13 == 0) )
    return (char)v9;
  if ( v11 )
  {
    if ( v13 )
      return (char)v9;
    goto LABEL_15;
  }
  if ( *(_BYTE *)(a2 + 16) == 54 )
  {
    if ( a5 && !byte_4FBEA80 )
      return (char)v9;
    v16 = 1;
  }
  else
  {
    v16 = a5;
  }
  v17 = 0;
  v18 = off_4CD4978[0];
  if ( off_4CD4978[0] )
  {
    v24 = v13;
    v19 = strlen(off_4CD4978[0]);
    v13 = v24;
    v17 = v19;
  }
  sub_1626100(a2, v18, v17, v13);
  sub_1CCABF0(2, v21);
  LOBYTE(v9) = sub_1CB4290(a1, a2, a4, v16);
  while ( 1 )
  {
    a2 = *(_QWORD *)(a2 + 8);
    if ( !a2 )
      break;
    while ( 1 )
    {
      v9 = (char **)sub_1648700(a2);
      if ( *((_BYTE *)v9 + 16) <= 0x17u )
        break;
      LOBYTE(v9) = sub_1CB4290(a1, (__int64)v9, a4, v16);
      a2 = *(_QWORD *)(a2 + 8);
      if ( !a2 )
        return (char)v9;
    }
  }
  return (char)v9;
}
