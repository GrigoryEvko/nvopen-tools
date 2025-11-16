// Function: sub_3785060
// Address: 0x3785060
//
unsigned __int8 *__fastcall sub_3785060(__int64 *a1, __int64 a2)
{
  __int64 v3; // r9
  __int64 v4; // rax
  __int64 v5; // rsi
  _QWORD *v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // r15
  unsigned __int8 *v9; // r12
  __int128 v11; // [rsp+0h] [rbp-60h] BYREF
  __int128 v12; // [rsp+10h] [rbp-50h] BYREF
  __int64 v13; // [rsp+20h] [rbp-40h] BYREF
  int v14; // [rsp+28h] [rbp-38h]

  HIWORD(v7) = 0;
  *(_QWORD *)&v11 = 0;
  DWORD2(v11) = 0;
  *(_QWORD *)&v12 = 0;
  DWORD2(v12) = 0;
  sub_377E430(a1, a2, (__int64 *)&v11, (__int64)&v12);
  v4 = *(_QWORD *)(a2 + 48);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = (_QWORD *)a1[1];
  LOWORD(v7) = *(_WORD *)v4;
  v8 = *(_QWORD *)(v4 + 8);
  v13 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v13, v5, 1);
  v14 = *(_DWORD *)(a2 + 72);
  v9 = sub_3406EB0(v6, 0x9Fu, (__int64)&v13, v7, v8, v3, v11, v12);
  if ( v13 )
    sub_B91220((__int64)&v13, v13);
  return v9;
}
