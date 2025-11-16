// Function: sub_1F5BD10
// Address: 0x1f5bd10
//
__int64 __fastcall sub_1F5BD10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 (*v6)(void); // rdx
  __int64 v7; // rax
  __int64 (*v8)(void); // rdx
  __int64 v9; // rax

  *(_QWORD *)(a1 + 232) = *(_QWORD *)(a2 + 40);
  v6 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v7 = 0;
  if ( v6 != sub_1D00B00 )
    v7 = v6();
  *(_QWORD *)(a1 + 240) = v7;
  v8 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 112LL);
  v9 = 0;
  if ( v8 != sub_1D00B10 )
    v9 = v8();
  *(_QWORD *)(a1 + 248) = v9;
  *(_QWORD *)(a1 + 256) = a2;
  *(_DWORD *)(a1 + 272) = 0;
  *(_DWORD *)(a1 + 296) = 0;
  *(_DWORD *)(a1 + 320) = 0;
  sub_1F5BB60((_DWORD *)a1, a2, (__int64)v8, a4, a5, a6);
  return 0;
}
