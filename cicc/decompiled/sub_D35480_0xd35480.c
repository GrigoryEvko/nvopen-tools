// Function: sub_D35480
// Address: 0xd35480
//
__int64 __fastcall sub_D35480(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  __int64 v6; // r8
  __int64 v7; // rdi
  _QWORD v9[2]; // [rsp+0h] [rbp-10h] BYREF

  v6 = *(_QWORD *)(a1 + 8);
  v9[0] = a1;
  v7 = *(_QWORD *)(a2 - 32);
  v9[1] = a2;
  return sub_D33610(v7, v6, (void (__fastcall *)(__int64, __int64))sub_D3A800, (__int64)v9, v6, a6);
}
