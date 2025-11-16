// Function: sub_30F4340
// Address: 0x30f4340
//
char __fastcall sub_30F4340(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  char *v4; // rdx
  char v5; // al
  __int64 v6; // r14
  __int64 *v7; // rax
  char result; // al
  __int64 *v9; // rbx
  __int64 v10; // r14
  __int64 *v11; // r15
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 *v14; // r14

  v3 = 0;
  v4 = *(char **)(a1 + 8);
  v5 = *v4;
  if ( (unsigned __int8)*v4 > 0x1Cu )
  {
    if ( v5 == 61 || v5 == 62 )
    {
      v3 = *((_QWORD *)v4 - 4);
    }
    else if ( v5 == 63 )
    {
      v3 = *(_QWORD *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
    }
  }
  v6 = *(_QWORD *)(a1 + 104);
  v7 = sub_DD8400(v6, v3);
  result = sub_DADE90(v6, (__int64)v7, a2);
  if ( !result )
  {
    v9 = *(__int64 **)(a1 + 24);
    v10 = 8LL * *(unsigned int *)(a1 + 32);
    v11 = &v9[(unsigned __int64)v10 / 8];
    v12 = v10 >> 3;
    v13 = v10 >> 5;
    if ( v13 )
    {
      v14 = &v9[4 * v13];
      while ( sub_30F4170(a1, *v9, a2) )
      {
        if ( !sub_30F4170(a1, v9[1], a2) )
          return v11 == v9 + 1;
        if ( !sub_30F4170(a1, v9[2], a2) )
          return v11 == v9 + 2;
        if ( !sub_30F4170(a1, v9[3], a2) )
          return v11 == v9 + 3;
        v9 += 4;
        if ( v9 == v14 )
        {
          v12 = v11 - v9;
          goto LABEL_18;
        }
      }
      return v11 == v9;
    }
LABEL_18:
    if ( v12 != 2 )
    {
      if ( v12 != 3 )
      {
        if ( v12 != 1 )
          return 1;
LABEL_29:
        result = sub_30F4170(a1, *v9, a2);
        if ( result )
          return result;
        return v11 == v9;
      }
      if ( !sub_30F4170(a1, *v9, a2) )
        return v11 == v9;
      ++v9;
    }
    if ( !sub_30F4170(a1, *v9, a2) )
      return v11 == v9;
    ++v9;
    goto LABEL_29;
  }
  return result;
}
