// Function: sub_3802130
// Address: 0x3802130
//
unsigned __int8 *__fastcall sub_3802130(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r9
  __int64 v6; // rsi
  _QWORD *v7; // r13
  __int128 *v8; // rcx
  __int64 v9; // r15
  unsigned int v10; // r12d
  unsigned __int8 *v11; // r12
  __int128 *v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+10h] [rbp-60h] BYREF
  int v15; // [rsp+18h] [rbp-58h]
  __int128 v16; // [rsp+20h] [rbp-50h] BYREF
  __int64 v17; // [rsp+30h] [rbp-40h] BYREF
  int v18; // [rsp+38h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 40);
  v15 = 0;
  DWORD2(v16) = 0;
  v14 = 0;
  v4 = *(_QWORD *)(v3 + 48);
  *(_QWORD *)&v16 = 0;
  sub_375E6F0(a1, *(_QWORD *)(v3 + 40), v4, (__int64)&v14, (__int64)&v16);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = *(_QWORD **)(a1 + 8);
  v8 = *(__int128 **)(a2 + 40);
  v9 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v10 = **(unsigned __int16 **)(a2 + 48);
  v17 = v6;
  if ( v6 )
  {
    v13 = v8;
    sub_B96E90((__int64)&v17, v6, 1);
    v8 = v13;
  }
  v18 = *(_DWORD *)(a2 + 72);
  v11 = sub_3406EB0(v7, 0x98u, (__int64)&v17, v10, v9, v5, *v8, v16);
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
  return v11;
}
