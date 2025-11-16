// Function: sub_29C12C0
// Address: 0x29c12c0
//
bool __fastcall sub_29C12C0(__int64 *a1)
{
  __int64 v1; // rdx
  unsigned __int8 **v2; // rsi
  unsigned __int8 **v3; // rax
  unsigned __int8 **v4; // rax

  v1 = *a1;
  v2 = (unsigned __int8 **)a1[1];
  if ( v2 != (unsigned __int8 **)*a1 )
  {
    while ( 1 )
    {
      v4 = (unsigned __int8 **)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
      {
        if ( (unsigned int)**((unsigned __int8 **)*v4 + 17) - 12 <= 1 )
          break;
        v1 = (unsigned __int64)(v4 + 1) | 4;
        v3 = (unsigned __int8 **)v1;
      }
      else
      {
        if ( (unsigned int)*v4[17] - 12 <= 1 )
          break;
        v3 = v4 + 18;
        v1 = (__int64)v3;
      }
      if ( v2 == v3 )
        return v2 != v3;
    }
  }
  v3 = (unsigned __int8 **)v1;
  return v2 != v3;
}
