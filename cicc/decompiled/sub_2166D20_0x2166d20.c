// Function: sub_2166D20
// Address: 0x2166d20
//
__int64 __fastcall sub_2166D20(__int64 a1)
{
  _QWORD *v1; // rsi
  _QWORD *v2; // rax
  unsigned int v3; // eax
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // rsi
  unsigned __int8 *v13[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v14[6]; // [rsp+10h] [rbp-30h] BYREF

  v1 = (_QWORD *)sub_1C7F370(byte_4FD17C0);
  sub_1F46490(a1, v1, 1, 1, 1u);
  if ( byte_4FD2160 )
  {
    v9 = (_QWORD *)sub_39623E0();
    sub_1F46490(a1, v9, 1, 1, 0);
    if ( !byte_4FD16E0 )
      goto LABEL_3;
  }
  else if ( !byte_4FD16E0 )
  {
    goto LABEL_3;
  }
  v13[0] = (unsigned __int8 *)v14;
  sub_2165CE0((__int64 *)v13, "\n\n*** Final LLVM Code input to ISel ***\n", (__int64)"");
  v11 = sub_16BA580((__int64)v13, (__int64)"\n\n*** Final LLVM Code input to ISel ***\n", v10);
  v12 = (_QWORD *)sub_15EA000(v11, v13);
  sub_1F46490(a1, v12, 1, 1, 0);
  if ( (_QWORD *)v13[0] != v14 )
    j_j___libc_free_0(v13[0], v14[0] + 1LL);
LABEL_3:
  v2 = (_QWORD *)sub_21BC8B0();
  sub_1F46490(a1, v2, 1, 1, 1u);
  v3 = sub_1F45DD0(a1);
  v4 = (_QWORD *)sub_21BE0C0(*(_QWORD *)(a1 + 208), v3);
  sub_1F46490(a1, v4, 1, 1, 1u);
  v5 = (_QWORD *)sub_22035C0();
  sub_1F46490(a1, v5, 1, 1, 0);
  v6 = (_QWORD *)sub_2205540(*(_QWORD *)(a1 + 208));
  sub_1F46490(a1, v6, 1, 1, 0);
  v7 = (_QWORD *)sub_21DC050();
  sub_1F46490(a1, v7, 1, 1, 1u);
  return 0;
}
