// Function: sub_39A3E90
// Address: 0x39a3e90
//
__int64 __fastcall sub_39A3E90(__int64 a1)
{
  __int64 v1; // rax

  v1 = sub_396DD80(*(_QWORD *)(a1 + 192));
  return sub_39A3E10(
           a1,
           a1 + 8,
           114,
           *(_QWORD *)(*(_QWORD *)(a1 + 208) + 248LL),
           *(_QWORD *)(*(_QWORD *)(v1 + 280) + 8LL));
}
