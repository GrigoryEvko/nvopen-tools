// Function: sub_28C2740
// Address: 0x28c2740
//
unsigned __int8 *__fastcall sub_28C2740(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 *v7; // rdi
  unsigned __int64 v8; // rax
  unsigned __int8 *v9; // r12
  __int64 v10; // rax
  __int64 *v11; // rdi
  __int64 v12; // r8
  unsigned __int64 v13; // r12
  const char *v14; // rdx
  __int64 *v16; // [rsp+8h] [rbp-418h]
  unsigned __int64 v17[2]; // [rsp+10h] [rbp-410h] BYREF
  _QWORD v18[2]; // [rsp+20h] [rbp-400h] BYREF
  unsigned __int64 v19[2]; // [rsp+30h] [rbp-3F0h] BYREF
  _QWORD v20[2]; // [rsp+40h] [rbp-3E0h] BYREF
  const char *v21[4]; // [rsp+50h] [rbp-3D0h] BYREF
  __int16 v22; // [rsp+70h] [rbp-3B0h]
  _BYTE v23[928]; // [rsp+80h] [rbp-3A0h] BYREF

  v6 = *(_QWORD *)(a1 + 8);
  v18[0] = a3;
  v18[1] = a2;
  v7 = *(__int64 **)(v6 + 24);
  v17[0] = (unsigned __int64)v18;
  v17[1] = 0x200000002LL;
  v8 = sub_DCD310(v7, 0xAu, (__int64)v17, a4, a3);
  v9 = sub_28C1F40(*(_QWORD *)(a1 + 8), v8, **(_QWORD **)(a1 + 16));
  if ( v9 )
  {
    v16 = sub_DA3860(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 24LL), a4);
    v20[1] = sub_DA3860(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 24LL), (__int64)v9);
    v10 = *(_QWORD *)(a1 + 8);
    v20[0] = v16;
    v11 = *(__int64 **)(v10 + 24);
    v19[1] = 0x200000002LL;
    v19[0] = (unsigned __int64)v20;
    v13 = sub_DCD310(v11, 0xAu, (__int64)v19, (__int64)v16, v12);
    sub_27C1C30(
      (__int64)v23,
      *(__int64 **)(*(_QWORD *)(a1 + 8) + 24LL),
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL),
      (__int64)"nary-reassociate",
      1);
    v9 = (unsigned __int8 *)sub_F8DB90(
                              (__int64)v23,
                              v13,
                              *(_QWORD *)(**(_QWORD **)(a1 + 16) + 8LL),
                              **(_QWORD **)(a1 + 16) + 24LL,
                              0);
    v21[0] = sub_BD5D20(**(_QWORD **)(a1 + 16));
    v21[2] = ".nary";
    v21[1] = v14;
    v22 = 773;
    sub_BD6B50(v9, v21);
    sub_27C20B0((__int64)v23);
    if ( (_QWORD *)v19[0] != v20 )
      _libc_free(v19[0]);
  }
  if ( (_QWORD *)v17[0] != v18 )
    _libc_free(v17[0]);
  return v9;
}
