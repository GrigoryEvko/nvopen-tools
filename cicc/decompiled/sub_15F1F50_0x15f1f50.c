// Function: sub_15F1F50
// Address: 0x15f1f50
//
__int64 __fastcall sub_15F1F50(__int64 a1, __int64 a2, int a3, __int64 a4, int a5, __int64 a6)
{
  int v7; // r13d
  int v8; // r8d
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 result; // rax

  v7 = a5 & 0xFFFFFFF;
  sub_1648CB0(a1, a2, (unsigned int)(a3 + 24));
  v8 = *(_DWORD *)(a1 + 20);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 20) = v7 | v8 & 0xF0000000;
  sub_157E9D0(a6 + 40, a1);
  v9 = *(_QWORD *)(a6 + 40);
  v10 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 32) = a6 + 40;
  v9 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a1 + 24) = v9 | v10 & 7;
  *(_QWORD *)(v9 + 8) = a1 + 24;
  result = *(_QWORD *)(a6 + 40) & 7LL;
  *(_QWORD *)(a6 + 40) = result | (a1 + 24);
  return result;
}
