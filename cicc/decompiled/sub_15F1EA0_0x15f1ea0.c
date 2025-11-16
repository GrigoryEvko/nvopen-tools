// Function: sub_15F1EA0
// Address: 0x15f1ea0
//
__int64 __fastcall sub_15F1EA0(__int64 a1, __int64 a2, int a3, __int64 a4, int a5, __int64 a6)
{
  int v7; // r13d
  __int64 result; // rax
  int v9; // r8d
  __int64 v10; // rcx

  v7 = a5 & 0xFFFFFFF;
  result = sub_1648CB0(a1, a2, (unsigned int)(a3 + 24));
  v9 = *(_DWORD *)(a1 + 20);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 20) = v7 | v9 & 0xF0000000;
  if ( a6 )
  {
    sub_157E9D0(*(_QWORD *)(a6 + 40) + 40LL, a1);
    v10 = *(_QWORD *)(a6 + 24);
    *(_QWORD *)(a1 + 32) = a6 + 24;
    v10 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 24) = v10 | *(_QWORD *)(a1 + 24) & 7LL;
    *(_QWORD *)(v10 + 8) = a1 + 24;
    result = *(_QWORD *)(a6 + 24) & 7LL;
    *(_QWORD *)(a6 + 24) = result | (a1 + 24);
  }
  return result;
}
