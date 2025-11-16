// Function: sub_27DC820
// Address: 0x27dc820
//
__int64 __fastcall sub_27DC820(__int64 a1, int a2)
{
  if ( a2 == -1 )
    a2 = qword_4FFDC28;
  *(_QWORD *)a1 = 0;
  *(_WORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 104) = a1 + 128;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_BYTE *)(a1 + 64) = 0;
  *(_BYTE *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 112) = 16;
  *(_DWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 124) = 1;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = a1 + 288;
  *(_QWORD *)(a1 + 272) = 16;
  *(_DWORD *)(a1 + 280) = 0;
  *(_BYTE *)(a1 + 284) = 1;
  *(_DWORD *)(a1 + 420) = a2;
  return a1 + 288;
}
