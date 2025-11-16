// Function: sub_ECD6D0
// Address: 0xecd6d0
//
__int64 __fastcall sub_ECD6D0(__int64 a1)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = &unk_49E4A30;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_WORD *)(a1 + 32) = 0;
  return 0;
}
