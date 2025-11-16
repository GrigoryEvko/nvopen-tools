// Function: sub_6E1A20
// Address: 0x6e1a20
//
__int64 __fastcall sub_6E1A20(__int64 a1)
{
  char v1; // al

  v1 = *(_BYTE *)(a1 + 8);
  if ( v1 == 1 )
    return a1 + 32;
  if ( v1 == 2 )
    return a1 + 48;
  if ( v1 )
    sub_721090(a1);
  return *(_QWORD *)(a1 + 24) + 76LL;
}
