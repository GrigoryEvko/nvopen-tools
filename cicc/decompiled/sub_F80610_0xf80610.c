// Function: sub_F80610
// Address: 0xf80610
//
__int64 __fastcall sub_F80610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // al
  __int64 v7; // rdx
  __int64 v9; // [rsp+0h] [rbp-10h] BYREF
  char v10; // [rsp+8h] [rbp-8h]
  unsigned __int8 v11; // [rsp+9h] [rbp-7h]

  v6 = *(_BYTE *)(a1 + 512);
  v7 = *(_QWORD *)a1;
  v11 = 0;
  v10 = v6;
  v9 = v7;
  sub_F7B020(a2, (__int64)&v9, v7, a4, a2, a6);
  return v11 ^ 1u;
}
