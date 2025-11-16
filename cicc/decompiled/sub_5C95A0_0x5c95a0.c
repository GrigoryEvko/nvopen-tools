// Function: sub_5C95A0
// Address: 0x5c95a0
//
__int64 __fastcall sub_5C95A0(__int64 **a1, int a2, __int64 a3)
{
  int v4; // r15d
  __int64 *v5; // rbx
  int v6; // r14d
  int v7; // eax
  __int64 v9; // [rsp+0h] [rbp-30h]

  if ( a1[5] )
  {
    v4 = 2;
    if ( a2 == 1 )
      return v9;
  }
  else
  {
    v4 = 1;
  }
  v5 = *a1;
  if ( *a1 )
  {
    v6 = 1;
    do
    {
      v7 = sub_8D2E30(v5[1]);
      if ( a2 == v4 )
      {
        if ( !v7 )
          return sub_6851C0(1619, a3);
        *((_BYTE *)v5 + 34) |= 8u;
        return v9;
      }
      if ( !a2 && v7 )
      {
        *((_BYTE *)v5 + 34) |= 8u;
        v6 = 0;
      }
      v5 = (__int64 *)*v5;
      ++v4;
    }
    while ( v5 );
    if ( !v6 )
      return v9;
  }
  if ( a2 )
    return sub_6851C0(1620, a3);
  else
    return sub_684B30(1621, a3);
}
