// Function: sub_254C190
// Address: 0x254c190
//
bool __fastcall sub_254C190(unsigned __int8 *a1, unsigned __int64 a2)
{
  int v3; // edx
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rdx

  if ( a2 < (unsigned __int64)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)] )
    return 0;
  v3 = *a1;
  if ( v3 == 40 )
  {
    v4 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v4 = -32;
    if ( v3 != 85 )
    {
      v4 = -96;
      if ( v3 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) != 0 )
  {
    v5 = sub_BD2BC0((__int64)a1);
    v7 = v5 + v6;
    if ( (a1[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v7 >> 4) )
        goto LABEL_15;
    }
    else if ( (unsigned int)((v7 - sub_BD2BC0((__int64)a1)) >> 4) )
    {
      if ( (a1[7] & 0x80u) != 0 )
      {
        v8 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        v9 = sub_BD2BC0((__int64)a1);
        v4 -= 32LL * (unsigned int)(*(_DWORD *)(v9 + v10 - 4) - v8);
        return a2 < (unsigned __int64)&a1[v4];
      }
LABEL_15:
      BUG();
    }
  }
  return a2 < (unsigned __int64)&a1[v4];
}
