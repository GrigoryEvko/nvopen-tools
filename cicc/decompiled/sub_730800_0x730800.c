// Function: sub_730800
// Address: 0x730800
//
_BOOL8 __fastcall sub_730800(__int64 a1)
{
  char v1; // al

  v1 = *(_BYTE *)(a1 + 48);
  if ( v1 == 2 )
    return *(_BYTE *)(*(_QWORD *)(a1 + 56) + 173LL) == 0;
  if ( v1 == 3 )
    return *(_BYTE *)(*(_QWORD *)(a1 + 56) + 24LL) == 0;
  return 0;
}
