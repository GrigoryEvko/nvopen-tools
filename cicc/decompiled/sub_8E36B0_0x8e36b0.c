// Function: sub_8E36B0
// Address: 0x8e36b0
//
__int64 __fastcall sub_8E36B0(__int64 a1)
{
  __int64 v1; // rdi
  char v2; // al

  while ( 1 )
  {
    while ( 1 )
    {
      v1 = sub_8D21F0(a1);
      v2 = *(_BYTE *)(v1 + 140);
      if ( v2 != 6 )
        break;
      a1 = sub_8D46C0(v1);
    }
    if ( v2 != 8 )
      break;
    a1 = sub_8D4050(v1);
  }
  return v1;
}
