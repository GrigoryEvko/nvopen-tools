// Function: sub_380CA20
// Address: 0x380ca20
//
unsigned __int8 *__fastcall sub_380CA20(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // r13
  __int64 v6; // r14
  __int128 *v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // r15
  unsigned int v10; // r12d
  __int64 v11; // r8
  unsigned int v12; // esi
  unsigned __int8 *v13; // r12
  __int128 v15; // [rsp-10h] [rbp-70h]
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int128 *v17; // [rsp+10h] [rbp-50h]
  __int64 v18; // [rsp+20h] [rbp-40h] BYREF
  int v19; // [rsp+28h] [rbp-38h]

  v3 = sub_380AAE0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD **)(a1 + 8);
  v6 = v3;
  v7 = *(__int128 **)(a2 + 40);
  v9 = v8;
  v10 = **(unsigned __int16 **)(a2 + 48);
  v11 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v18 = v4;
  if ( v4 )
  {
    v16 = v11;
    v17 = v7;
    sub_B96E90((__int64)&v18, v4, 1);
    v11 = v16;
    v7 = v17;
  }
  *((_QWORD *)&v15 + 1) = v9;
  *(_QWORD *)&v15 = v6;
  v12 = *(_DWORD *)(a2 + 24);
  v19 = *(_DWORD *)(a2 + 72);
  v13 = sub_3406EB0(v5, v12, (__int64)&v18, v10, v11, (__int64)&v18, *v7, v15);
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return v13;
}
