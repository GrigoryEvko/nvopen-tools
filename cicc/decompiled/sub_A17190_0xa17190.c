// Function: sub_A17190
// Address: 0xa17190
//
__int64 __fastcall sub_A17190(unsigned __int8 *a1)
{
  int v1; // edx
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx

  v1 = *a1;
  if ( v1 == 40 )
  {
    v2 = 32LL * (unsigned int)sub_B491D0(a1);
  }
  else
  {
    v2 = 0;
    if ( v1 != 85 )
    {
      v2 = 64;
      if ( v1 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_10;
  v3 = sub_BD2BC0(a1);
  v5 = v3 + v4;
  if ( (a1[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v5 >> 4) )
LABEL_15:
      BUG();
LABEL_10:
    v9 = 0;
    return (32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v2 - v9) >> 5;
  }
  if ( !(unsigned int)((v5 - sub_BD2BC0(a1)) >> 4) )
    goto LABEL_10;
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_15;
  v6 = *(_DWORD *)(sub_BD2BC0(a1) + 8);
  if ( (a1[7] & 0x80u) == 0 )
    BUG();
  v7 = sub_BD2BC0(a1);
  v9 = 32LL * (unsigned int)(*(_DWORD *)(v7 + v8 - 4) - v6);
  return (32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v2 - v9) >> 5;
}
