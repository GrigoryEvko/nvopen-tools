// Function: sub_24E54B0
// Address: 0x24e54b0
//
unsigned __int8 *__fastcall sub_24E54B0(unsigned __int8 *a1)
{
  int v1; // edx
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // rdx

  v1 = *a1;
  if ( v1 == 40 )
  {
    v2 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v2 = -32;
    if ( v1 != 85 )
    {
      v2 = -96;
      if ( v1 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) != 0 )
  {
    v3 = sub_BD2BC0((__int64)a1);
    v5 = v3 + v4;
    if ( (a1[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v5 >> 4) )
        return &a1[v2];
    }
    else
    {
      if ( !(unsigned int)((v5 - sub_BD2BC0((__int64)a1)) >> 4) )
        return &a1[v2];
      if ( (a1[7] & 0x80u) != 0 )
      {
        v6 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        v7 = sub_BD2BC0((__int64)a1);
        v2 -= 32LL * (unsigned int)(*(_DWORD *)(v7 + v8 - 4) - v6);
        return &a1[v2];
      }
    }
    BUG();
  }
  return &a1[v2];
}
