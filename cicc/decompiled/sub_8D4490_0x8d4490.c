// Function: sub_8D4490
// Address: 0x8d4490
//
__int64 __fastcall sub_8D4490(__int64 a1)
{
  __int64 v1; // r8
  char v2; // al

  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  v1 = 1;
  do
  {
    v1 *= *(_QWORD *)(a1 + 176);
    do
    {
      a1 = *(_QWORD *)(a1 + 160);
      v2 = *(_BYTE *)(a1 + 140);
    }
    while ( v2 == 12 );
  }
  while ( v2 == 8 );
  return v1;
}
