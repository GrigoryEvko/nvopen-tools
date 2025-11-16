// Function: sub_2D19D50
// Address: 0x2d19d50
//
__int64 __fastcall sub_2D19D50(__int64 a1, __int64 a2)
{
  char v2; // al
  char v4[2]; // [rsp+Eh] [rbp-2h] BYREF

  v2 = *(_BYTE *)(a1 + 170);
  v4[0] = *(_BYTE *)(a1 + 169);
  v4[1] = v2;
  return sub_2D19B20(v4, a2);
}
