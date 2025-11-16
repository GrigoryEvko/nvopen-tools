// Function: sub_3567760
// Address: 0x3567760
//
__int64 __fastcall sub_3567760(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  *(_QWORD *)(a1 + 8) = a6;
  *(_QWORD *)a1 = a2 | 4;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 24) = a5;
  *(_QWORD *)(a1 + 32) = a3;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = a1 + 72;
  *(_QWORD *)(a1 + 96) = a1 + 72;
  *(_QWORD *)(a1 + 104) = 0;
  return a1 + 72;
}
