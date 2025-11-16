// Function: sub_168E6A0
// Address: 0x168e6a0
//
__int64 __fastcall sub_168E6A0(__int64 a1, __int64 a2, __int64 a3)
{
  sub_38DCAE0();
  *(_QWORD *)(a1 + 264) = a3;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)a1 = &unk_49EE5B0;
  *(_QWORD *)(a1 + 288) = 0x1000000000LL;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_DWORD *)(a1 + 328) = 0;
  return 0x1000000000LL;
}
