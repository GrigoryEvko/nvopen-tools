// Function: sub_C6D380
// Address: 0xc6d380
//
__int64 __fastcall sub_C6D380(__int64 a1, unsigned __int16 *a2, unsigned __int64 a3)
{
  __int64 result; // rax
  _QWORD v4[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v5; // [rsp+10h] [rbp-20h]

  sub_C6D250((__int64)v4, a2, a3);
  *(_WORD *)a1 = 8;
  *(_QWORD *)(a1 + 8) = v4[0];
  *(_QWORD *)(a1 + 16) = v4[1];
  result = v5;
  *(_QWORD *)(a1 + 24) = v5;
  return result;
}
