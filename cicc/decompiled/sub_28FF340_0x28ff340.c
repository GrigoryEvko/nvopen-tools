// Function: sub_28FF340
// Address: 0x28ff340
//
__int64 __fastcall sub_28FF340(__int64 a1)
{
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = a1 + 24;
  *(_QWORD *)(a1 + 48) = a1 + 24;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 8) = 201342976;
  *(_QWORD *)a1 = 0xC000020400000LL;
  return a1;
}
