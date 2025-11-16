// Function: sub_8D49A0
// Address: 0x8d49a0
//
__int64 __fastcall sub_8D49A0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rdi
  unsigned __int8 v3; // al

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = sub_8D21F0(v1);
      v3 = *(_BYTE *)(v2 + 140);
      if ( v3 > 8u )
      {
        if ( v3 != 13 )
          return v1;
      }
      else if ( v3 <= 5u )
      {
        return v1;
      }
      v1 = sub_8D48B0(v2, 0);
    }
    while ( v1 );
  }
  return 0;
}
