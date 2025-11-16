// Function: sub_AA4C30
// Address: 0xaa4c30
//
void __fastcall sub_AA4C30(__int64 a1, char a2)
{
  char v2; // al

  v2 = *(_BYTE *)(a1 + 40);
  if ( a2 )
  {
    if ( !v2 )
      sub_AA45D0(a1);
  }
  else if ( v2 )
  {
    sub_AA4B40(a1);
  }
}
