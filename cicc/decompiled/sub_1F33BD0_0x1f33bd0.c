// Function: sub_1F33BD0
// Address: 0x1f33bd0
//
__int64 __fastcall sub_1F33BD0(__int64 a1, __int64 a2, char a3, __int64 a4, char a5, int a6)
{
  __int64 (*v10)(void); // rdx
  __int64 v11; // rax
  __int64 (*v12)(void); // rdx
  __int64 v13; // rax
  __int64 result; // rax

  *(_QWORD *)(a1 + 40) = a2;
  v10 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v11 = 0;
  if ( v10 != sub_1D00B00 )
  {
    v11 = v10();
    a2 = *(_QWORD *)(a1 + 40);
  }
  *(_QWORD *)a1 = v11;
  v12 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 112LL);
  v13 = 0;
  if ( v12 != sub_1D00B10 )
  {
    v13 = v12();
    a2 = *(_QWORD *)(a1 + 40);
  }
  *(_QWORD *)(a1 + 8) = v13;
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 40);
  result = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 16) = a4;
  *(_DWORD *)(a1 + 52) = a6;
  *(_BYTE *)(a1 + 49) = a5;
  *(_BYTE *)(a1 + 48) = a3;
  *(_QWORD *)(a1 + 24) = result;
  return result;
}
