// Function: sub_AF4460
// Address: 0xaf4460
//
__int64 __fastcall sub_AF4460(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v2; // r12d
  unsigned __int64 *v3; // r13
  unsigned __int64 *v4; // rbx
  unsigned __int64 *v6; // [rsp+8h] [rbp-28h] BYREF

  LOBYTE(v1) = sub_AF4230(a1);
  v2 = v1;
  if ( !(_BYTE)v1 )
    return v2;
  v3 = *(unsigned __int64 **)(a1 + 24);
  v4 = *(unsigned __int64 **)(a1 + 16);
  if ( (unsigned int)(v3 - v4) )
  {
    v6 = *(unsigned __int64 **)(a1 + 16);
    if ( v4 != v3 )
    {
      while ( *v4 != 159 )
      {
        v4 += (unsigned int)sub_AF4160(&v6);
        v6 = v4;
        if ( v3 == v4 )
          return 0;
      }
      return v2;
    }
  }
  return 0;
}
