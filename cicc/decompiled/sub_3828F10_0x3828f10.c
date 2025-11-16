// Function: sub_3828F10
// Address: 0x3828f10
//
unsigned __int8 *__fastcall sub_3828F10(__int64 a1, __int64 a2)
{
  unsigned __int64 *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r9
  __int64 v6; // rsi
  _QWORD *v7; // r13
  __int64 v8; // r15
  unsigned int v9; // r12d
  unsigned __int8 *v10; // r12
  __int128 v12; // [rsp+0h] [rbp-60h] BYREF
  __int128 v13; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+20h] [rbp-40h] BYREF
  int v15; // [rsp+28h] [rbp-38h]

  v3 = *(unsigned __int64 **)(a2 + 40);
  DWORD2(v12) = 0;
  DWORD2(v13) = 0;
  *(_QWORD *)&v12 = 0;
  v4 = v3[1];
  *(_QWORD *)&v13 = 0;
  sub_375E510(a1, *v3, v4, (__int64)&v12, (__int64)&v13);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = *(_QWORD **)(a1 + 8);
  v8 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v9 = **(unsigned __int16 **)(a2 + 48);
  v14 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v14, v6, 1);
  v15 = *(_DWORD *)(a2 + 72);
  v10 = sub_3406EB0(v7, 0xA9u, (__int64)&v14, v9, v8, v5, v12, v13);
  if ( v14 )
    sub_B91220((__int64)&v14, v14);
  return v10;
}
