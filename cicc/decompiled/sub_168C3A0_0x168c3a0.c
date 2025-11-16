// Function: sub_168C3A0
// Address: 0x168c3a0
//
__int64 __fastcall sub_168C3A0(__int64 a1)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = a1 + 16;
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}
