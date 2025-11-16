// Function: sub_2FE3D70
// Address: 0x2fe3d70
//
bool __fastcall sub_2FE3D70(__int64 a1, __int64 a2)
{
  char v2; // r8
  bool result; // al
  __int64 v4; // [rsp+8h] [rbp-18h] BYREF

  v4 = sub_B2D7E0(a2, "no-jump-tables", 0xEu);
  v2 = sub_A72A30(&v4);
  result = 0;
  if ( !v2 )
  {
    result = 1;
    if ( (*(_BYTE *)(a1 + 7217) & 0xFB) != 0 )
      return (*(_BYTE *)(a1 + 7216) & 0xFB) == 0;
  }
  return result;
}
