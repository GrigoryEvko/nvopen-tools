// Function: sub_E3FC50
// Address: 0xe3fc50
//
__int64 __fastcall sub_E3FC50(__int64 a1)
{
  *(_QWORD *)(a1 + 16) = 0;
  *(_BYTE *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_QWORD *)a1 = &unk_49DB390;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  return a1 + 24;
}
