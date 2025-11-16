// Function: sub_2163460
// Address: 0x2163460
//
void __fastcall sub_2163460(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rbx
  int v9; // r12d

  v5 = *(_QWORD *)(a2 + 32);
  v6 = v5 + 40LL * a4;
  v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 24) + 56LL) + 56LL);
  v8 = 40LL * (a4 + 1);
  v9 = *(_DWORD *)(*(_QWORD *)(v7 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v7 + 32) + *(_DWORD *)(v6 + 24)))
     + *(_QWORD *)(v5 + v8 + 24);
  sub_1E31400((char *)v6, 2, 0, 0, 0, 0, 0, 0);
  sub_1E313C0(*(_QWORD *)(a2 + 32) + v8, v9);
}
