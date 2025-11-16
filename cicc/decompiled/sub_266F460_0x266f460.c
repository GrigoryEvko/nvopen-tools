// Function: sub_266F460
// Address: 0x266f460
//
__int64 __fastcall sub_266F460(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9

  v7 = a1 + 8;
  *(_DWORD *)(v7 - 8) = *(_DWORD *)(a2 + 104);
  sub_C8CD80(v7, a1 + 40, a2 + 112, a4, a5, a6);
  sub_C8CD80(a1 + 56, a1 + 88, a2 + 160, v8, v9, v10);
  return a1;
}
