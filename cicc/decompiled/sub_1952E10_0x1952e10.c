// Function: sub_1952E10
// Address: 0x1952e10
//
__int64 __fastcall sub_1952E10(__int64 a1, int a2)
{
  if ( a2 == -1 )
    a2 = dword_4FB02A0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_WORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = a1 + 96;
  *(_QWORD *)(a1 + 72) = a1 + 96;
  *(_QWORD *)(a1 + 80) = 16;
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_DWORD *)(a1 + 248) = 0;
  *(_DWORD *)(a1 + 256) = a2;
  return a1 + 96;
}
