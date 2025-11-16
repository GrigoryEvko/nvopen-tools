// Function: sub_15EAAD0
// Address: 0x15eaad0
//
__int64 __fastcall sub_15EAAD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, char a6, unsigned int a7)
{
  __int64 v10; // rax

  v10 = sub_1646BA0(a2, 0);
  sub_1648CB0(a1, v10, 20);
  *(_QWORD *)(a1 + 24) = a1 + 40;
  sub_15EA590((__int64 *)(a1 + 24), *(_BYTE **)a3, *(_QWORD *)a3 + *(_QWORD *)(a3 + 8));
  *(_QWORD *)(a1 + 56) = a1 + 72;
  sub_15EA590((__int64 *)(a1 + 56), *(_BYTE **)a4, *(_QWORD *)a4 + *(_QWORD *)(a4 + 8));
  *(_QWORD *)(a1 + 88) = a2;
  *(_BYTE *)(a1 + 97) = a6;
  *(_BYTE *)(a1 + 96) = a5;
  *(_DWORD *)(a1 + 100) = a7;
  return a7;
}
