// Function: sub_5F2700
// Address: 0x5f2700
//
__int64 __fastcall sub_5F2700(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 result; // rax
  __int64 v13[3]; // [rsp+8h] [rbp-18h] BYREF

  v7 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v13[0] = v7;
  sub_6D6DB0(a1, v7);
  *(_BYTE *)(a2 + 177) = 1;
  result = sub_724E50(v13, v7, v8, v9, v10, v11);
  *(_BYTE *)(a2 + 174) |= 0x10u;
  *(_QWORD *)(a2 + 184) = result;
  return result;
}
