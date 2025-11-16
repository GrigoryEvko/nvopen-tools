// Function: sub_B13130
// Address: 0xb13130
//
__int64 __fastcall sub_B13130(__int64 a1)
{
  char v1; // al

  v1 = *(_BYTE *)(a1 + 32);
  if ( !v1 )
    return sub_B13070(a1);
  if ( v1 != 1 )
    BUG();
  return sub_B130B0(a1);
}
