// Function: sub_2FC8820
// Address: 0x2fc8820
//
__int64 __fastcall sub_2FC8820(__int64 a1, __int64 a2)
{
  bool v2; // zf
  __int64 result; // rax

  v2 = (_DWORD)qword_5026048 == 3;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  result = a1 + 128;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = a1 + 128;
  *(_QWORD *)(a1 + 120) = 0;
  if ( !v2 )
    BUG();
  return result;
}
