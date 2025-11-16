// Function: sub_AF4590
// Address: 0xaf4590
//
bool __fastcall sub_AF4590(__int64 a1)
{
  bool result; // al
  unsigned __int64 *v2; // r12
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 *i; // [rsp+8h] [rbp-28h] BYREF

  result = sub_AF4230(a1);
  if ( result )
  {
    v2 = *(unsigned __int64 **)(a1 + 24);
    v3 = *(unsigned __int64 **)(a1 + 16);
    if ( (unsigned int)(v2 - v3) )
    {
      v4 = *(unsigned __int64 **)(a1 + 16);
      if ( *v3 == 4101 )
      {
        result = 0;
        if ( v3[1] )
          return result;
        v3 += (unsigned int)sub_AF4160(&v4);
        v4 = v3;
      }
      for ( i = v3; v2 != v3; i = v3 )
      {
        if ( *v3 == 4101 )
          break;
        v3 += (unsigned int)sub_AF4160(&i);
      }
      return v2 == v3;
    }
  }
  return result;
}
