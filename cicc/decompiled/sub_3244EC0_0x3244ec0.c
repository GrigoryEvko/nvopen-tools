// Function: sub_3244EC0
// Address: 0x3244ec0
//
__int64 __fastcall sub_3244EC0(__int64 *a1, __int64 a2, unsigned int a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // r8
  __int64 v6; // r9
  int v8; // [rsp+Ah] [rbp-26h] BYREF
  __int16 v9; // [rsp+Eh] [rbp-22h]

  v4 = sub_31DF6E0(*a1);
  v8 = v4;
  v9 = WORD2(v4);
  return sub_3216C50(a2, (__int64)&v8, a1 + 13, a3, v5, v6);
}
