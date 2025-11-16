// Function: sub_15F4D40
// Address: 0x15f4d40
//
bool __fastcall sub_15F4D40(__int64 a1)
{
  char v1; // al

  v1 = *(_BYTE *)(a1 + 8);
  if ( v1 == 16 )
    v1 = *(_BYTE *)(**(_QWORD **)(a1 + 16) + 8LL);
  return (unsigned __int8)(v1 - 1) <= 5u;
}
