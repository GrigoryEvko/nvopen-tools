// Function: sub_38CF6F0
// Address: 0x38cf6f0
//
unsigned __int64 __fastcall sub_38CF6F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // r8

  v4 = *(unsigned int *)(a1 + 480);
  v5 = a3 & (v4 - 1);
  v6 = v5 + a4;
  if ( *(_BYTE *)(a2 + 48) )
  {
    v7 = 0;
    if ( v4 != v6 )
    {
      v7 = 2 * v4 - v6;
      if ( v4 > v6 )
        return v4 - v6;
    }
    return v7;
  }
  else if ( v5 && v4 < v6 )
  {
    return v4 - v5;
  }
  else
  {
    return 0;
  }
}
