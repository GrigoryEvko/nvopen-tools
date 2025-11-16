// Function: sub_30D6600
// Address: 0x30d6600
//
char *__fastcall sub_30D6600(unsigned __int8 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  int v7; // edx
  int v8; // r15d
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // r14
  const char *v21; // rax
  _QWORD v23[2]; // [rsp+20h] [rbp-50h] BYREF
  const char *v24; // [rsp+30h] [rbp-40h] BYREF
  __int64 v25; // [rsp+38h] [rbp-38h]

  v23[0] = a4;
  v23[1] = a5;
  if ( !a2 )
  {
    LOBYTE(v25) = 1;
    return "indirect call";
  }
  if ( (unsigned __int8)sub_B2D610(a2, 49) )
  {
    LOBYTE(v25) = 1;
    return "unsplited coroutine call";
  }
  v6 = sub_B2BEC0(a2);
  v7 = *a1;
  v8 = *(_DWORD *)(v6 + 4);
  if ( v7 == 40 )
  {
    v9 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v9 = 0;
    if ( v7 != 85 )
    {
      v9 = 64;
      if ( v7 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_29;
  v10 = sub_BD2BC0((__int64)a1);
  v12 = v10 + v11;
  if ( (a1[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v12 >> 4) )
LABEL_47:
      BUG();
LABEL_29:
    v16 = 0;
    goto LABEL_17;
  }
  if ( !(unsigned int)((v12 - sub_BD2BC0((__int64)a1)) >> 4) )
    goto LABEL_29;
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_47;
  v13 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
  if ( (a1[7] & 0x80u) == 0 )
    BUG();
  v14 = sub_BD2BC0((__int64)a1);
  v16 = 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
LABEL_17:
  v17 = 0;
  v18 = (32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v9 - v16) >> 5;
  v19 = (unsigned int)v18;
  if ( !(_DWORD)v18 )
  {
LABEL_21:
    if ( (unsigned __int8)sub_A73ED0((_QWORD *)a1 + 9, 3) || (unsigned __int8)sub_B49560((__int64)a1, 3) )
    {
      v24 = (const char *)*((_QWORD *)a1 + 9);
      if ( !(unsigned __int8)sub_A73ED0(&v24, 31) )
      {
        v21 = sub_30D6380(a2);
        if ( !v21 )
        {
          v24 = 0;
          LOBYTE(v25) = 1;
          return (char *)v24;
        }
        goto LABEL_25;
      }
    }
    else
    {
      v20 = sub_B491C0((__int64)a1);
      if ( !(unsigned __int8)sub_30D1240(v20, a2, a3, (__int64)v23) )
      {
        v21 = "conflicting attributes";
LABEL_25:
        v24 = v21;
        LOBYTE(v25) = 1;
        return (char *)v24;
      }
      if ( (unsigned __int8)sub_B2D610(v20, 48) )
      {
        LOBYTE(v25) = 1;
        return "optnone attribute";
      }
      if ( !(unsigned __int8)sub_B2F060(v20) && (unsigned __int8)sub_B2F060(a2) )
      {
        LOBYTE(v25) = 1;
        return "nullptr definitions incompatible";
      }
      if ( (unsigned __int8)sub_B2F6B0(a2) )
      {
        LOBYTE(v25) = 1;
        return "interposable";
      }
      if ( (unsigned __int8)sub_B2D610(a2, 31) )
      {
        LOBYTE(v25) = 1;
        return "noinline function attribute";
      }
      if ( !(unsigned __int8)sub_A73ED0((_QWORD *)a1 + 9, 31) && !(unsigned __int8)sub_B49560((__int64)a1, 31) )
      {
        LOBYTE(v25) = 0;
        return (char *)v24;
      }
    }
    LOBYTE(v25) = 1;
    return "noinline call site attribute";
  }
  while ( !(unsigned __int8)sub_B49B80((__int64)a1, v17, 81)
       || *(_DWORD *)(*(_QWORD *)(*(_QWORD *)&a1[32 * (v17 - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))] + 8LL) + 8LL) >> 8 == v8 )
  {
    if ( ++v17 == v19 )
      goto LABEL_21;
  }
  LOBYTE(v25) = 1;
  return "byval arguments without alloca address space";
}
