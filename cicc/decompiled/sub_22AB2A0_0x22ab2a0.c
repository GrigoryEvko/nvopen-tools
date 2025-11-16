// Function: sub_22AB2A0
// Address: 0x22ab2a0
//
__int64 __fastcall sub_22AB2A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax

  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x100000000LL;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  v5 = sub_BC0510(a4, &unk_4FDB6A0, a3);
  sub_22AA8A0(a1, a3, v5 + 8);
  return a1;
}
