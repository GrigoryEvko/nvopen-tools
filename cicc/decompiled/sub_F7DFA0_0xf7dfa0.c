// Function: sub_F7DFA0
// Address: 0xf7dfa0
//
__int64 __fastcall sub_F7DFA0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  _BYTE v6[32]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v7; // [rsp+20h] [rbp-20h]

  v3 = *(_QWORD *)(a2 + 32);
  v7 = 257;
  v4 = sub_AD64C0(v3, 1, 0);
  return sub_B33D80(a1 + 520, v4, (__int64)v6);
}
