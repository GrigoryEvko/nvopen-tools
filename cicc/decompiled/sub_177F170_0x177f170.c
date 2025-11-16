// Function: sub_177F170
// Address: 0x177f170
//
_QWORD *__fastcall sub_177F170(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 *a5)
{
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // r12
  __int64 v11; // rdi
  unsigned __int64 *v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  bool v16; // zf
  __int64 v17; // rsi
  __int64 v18; // rsi
  unsigned __int8 *v19; // rsi
  _QWORD v22[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v23; // [rsp+20h] [rbp-40h]

  v7 = sub_15F24E0(a4);
  v23 = 257;
  v8 = sub_15FB440(16, a2, a3, (__int64)v22, 0);
  v9 = *(_QWORD *)(a1 + 32);
  v10 = (_QWORD *)v8;
  if ( v9 )
    sub_1625C10(v8, 3, v9);
  sub_15F2440((__int64)v10, v7);
  v11 = *(_QWORD *)(a1 + 8);
  if ( v11 )
  {
    v12 = *(unsigned __int64 **)(a1 + 16);
    sub_157E9D0(v11 + 40, (__int64)v10);
    v13 = v10[3];
    v14 = *v12;
    v10[4] = v12;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    v10[3] = v14 | v13 & 7;
    *(_QWORD *)(v14 + 8) = v10 + 3;
    *v12 = *v12 & 7 | (unsigned __int64)(v10 + 3);
  }
  sub_164B780((__int64)v10, a5);
  v16 = *(_QWORD *)(a1 + 80) == 0;
  v22[0] = v10;
  if ( v16 )
    sub_4263D6(v10, a5, v15);
  (*(void (__fastcall **)(__int64, _QWORD *))(a1 + 88))(a1 + 64, v22);
  v17 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v22[0] = *(_QWORD *)a1;
    sub_1623A60((__int64)v22, v17, 2);
    v18 = v10[6];
    if ( v18 )
      sub_161E7C0((__int64)(v10 + 6), v18);
    v19 = (unsigned __int8 *)v22[0];
    v10[6] = v22[0];
    if ( v19 )
      sub_1623210((__int64)v22, v19, (__int64)(v10 + 6));
  }
  return v10;
}
