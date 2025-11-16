// Function: sub_87E3B0
// Address: 0x87e3b0
//
__int64 __fastcall sub_87E3B0(__int64 a1)
{
  __int64 result; // rax

  *(_QWORD *)a1 = 0;
  *(_WORD *)(a1 + 64) &= 0xE000u;
  result = *(_QWORD *)&dword_4F077C8;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = result;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = -1;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 96) = 0;
  return result;
}
