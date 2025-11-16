// Function: sub_DA2D20
// Address: 0xda2d20
//
_QWORD *__fastcall sub_DA2D20(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v8; // [rsp+0h] [rbp-30h] BYREF
  char v9; // [rsp+8h] [rbp-28h]

  v4 = 16LL * a4 + sub_AE4AC0(*(_QWORD *)(a1 + 8), a3) + 24;
  v5 = *(_QWORD *)v4;
  LOBYTE(v4) = *(_BYTE *)(v4 + 8);
  v8 = v5;
  v9 = v4;
  v6 = sub_CA1930(&v8);
  return sub_DA2C50(a1, a2, v6, 0);
}
