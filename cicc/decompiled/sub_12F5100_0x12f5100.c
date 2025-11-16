// Function: sub_12F5100
// Address: 0x12f5100
//
__int64 __fastcall sub_12F5100(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, _QWORD *a5)
{
  unsigned int v7; // eax
  unsigned int v9; // r12d

  sub_1C3E900();
  v7 = sub_1BF83F0();
  if ( !(_BYTE)v7 )
    return sub_12F4060(a1, a2, a3, a4, a5);
  v9 = v7;
  sub_1C3E9C0(a4);
  return v9;
}
