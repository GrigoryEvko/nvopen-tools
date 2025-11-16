// Function: sub_B2B9F0
// Address: 0xb2b9f0
//
void __fastcall sub_B2B9F0(__int64 a1, char a2)
{
  char v2; // al

  v2 = *(_BYTE *)(a1 + 128);
  if ( a2 )
  {
    if ( !v2 )
      sub_B2B950(a1);
  }
  else if ( v2 )
  {
    sub_B2B9A0(a1);
  }
}
