// Function: sub_A7E990
// Address: 0xa7e990
//
unsigned __int8 *__fastcall sub_A7E990(unsigned __int8 *a1)
{
  int v1; // edx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r13

  v1 = *a1;
  if ( v1 == 40 )
  {
    sub_B491D0(a1);
  }
  else if ( v1 != 85 && v1 != 34 )
  {
    BUG();
  }
  if ( (a1[7] & 0x80u) != 0 )
  {
    v2 = sub_BD2BC0(a1);
    v4 = v2 + v3;
    if ( (a1[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v4 >> 4) )
        return &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    }
    else
    {
      if ( !(unsigned int)((v4 - sub_BD2BC0(a1)) >> 4) )
        return &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      if ( (a1[7] & 0x80u) != 0 )
      {
        sub_BD2BC0(a1);
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        sub_BD2BC0(a1);
        return &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      }
    }
    BUG();
  }
  return &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
}
