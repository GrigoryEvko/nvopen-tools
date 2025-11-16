// Function: sub_25093E0
// Address: 0x25093e0
//
__int64 __fastcall sub_25093E0(unsigned __int8 **a1)
{
  int v1; // eax
  __int64 v3; // rbx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  int v9; // r13d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx

  v1 = *((_DWORD *)a1 + 4);
  if ( v1 )
    return (unsigned int)(v1 - 1);
  if ( sub_B491E0((__int64)*a1) )
    return (unsigned int)(*((_DWORD *)a1 + 4) - 1);
  v3 = (__int64)*a1;
  v4 = **a1;
  if ( v4 == 40 )
  {
    v5 = 32LL * (unsigned int)sub_B491D0(v3);
  }
  else
  {
    v5 = 0;
    if ( v4 != 85 )
    {
      v5 = 64;
      if ( v4 != 34 )
        BUG();
    }
  }
  if ( *(char *)(v3 + 7) < 0 )
  {
    v6 = sub_BD2BC0(v3);
    v8 = v6 + v7;
    if ( *(char *)(v3 + 7) >= 0 )
    {
      if ( (unsigned int)(v8 >> 4) )
        goto LABEL_20;
    }
    else if ( (unsigned int)((v8 - sub_BD2BC0(v3)) >> 4) )
    {
      if ( *(char *)(v3 + 7) < 0 )
      {
        v9 = *(_DWORD *)(sub_BD2BC0(v3) + 8);
        if ( *(char *)(v3 + 7) >= 0 )
          BUG();
        v10 = sub_BD2BC0(v3);
        v12 = 32LL * (unsigned int)(*(_DWORD *)(v10 + v11 - 4) - v9);
        return (32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF) - 32 - v5 - v12) >> 5;
      }
LABEL_20:
      BUG();
    }
  }
  v12 = 0;
  return (32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF) - 32 - v5 - v12) >> 5;
}
