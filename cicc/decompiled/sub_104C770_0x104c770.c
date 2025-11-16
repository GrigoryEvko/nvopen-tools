// Function: sub_104C770
// Address: 0x104c770
//
__int64 __fastcall sub_104C770(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 140) = 0;
  *(_QWORD *)(a1 + 56) = 0x600000000LL;
  *(_QWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 136) = 0;
  v3 = *(_DWORD *)(a3 + 92);
  *(_QWORD *)(a1 + 128) = a3;
  *(_DWORD *)(a1 + 144) = v3;
  sub_B29120(a1);
  return a1;
}
