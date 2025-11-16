// Function: sub_37874C0
// Address: 0x37874c0
//
unsigned __int8 *__fastcall sub_37874C0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // r9
  __int64 v5; // rax
  __int64 v6; // rsi
  _QWORD *v7; // r13
  __int64 v8; // r15
  unsigned int v9; // r12d
  unsigned __int8 *v10; // r12
  __int128 v12; // [rsp+0h] [rbp-60h] BYREF
  __int128 v13; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+20h] [rbp-40h] BYREF
  int v15; // [rsp+28h] [rbp-38h]

  *(_QWORD *)&v12 = 0;
  DWORD2(v12) = 0;
  *(_QWORD *)&v13 = 0;
  DWORD2(v13) = 0;
  sub_377A7C0(a1, a2, (__int64)&v12, (__int64)&v13, a3);
  v5 = *(_QWORD *)(a2 + 48);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = (_QWORD *)a1[1];
  v8 = *(_QWORD *)(v5 + 8);
  v9 = **(unsigned __int16 **)(a2 + 48);
  v14 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v14, v6, 1);
  v15 = *(_DWORD *)(a2 + 72);
  v10 = sub_3406EB0(v7, 0x9Fu, (__int64)&v14, v9, v8, v4, v12, v13);
  if ( v14 )
    sub_B91220((__int64)&v14, v14);
  return v10;
}
