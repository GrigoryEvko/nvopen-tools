// Function: sub_3794AB0
// Address: 0x3794ab0
//
unsigned __int8 *__fastcall sub_3794AB0(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r14
  int v10; // r9d
  __int64 v11; // rdx
  __int64 v12; // r15
  _QWORD *v13; // r11
  __int64 v14; // rcx
  __int64 v15; // r8
  unsigned int v16; // esi
  unsigned __int8 *v17; // r12
  __int128 v19; // [rsp-20h] [rbp-90h]
  __int128 v20; // [rsp-10h] [rbp-80h]
  __int64 v21; // [rsp+8h] [rbp-68h]
  int v22; // [rsp+14h] [rbp-5Ch]
  __int64 v23; // [rsp+18h] [rbp-58h]
  _QWORD *v24; // [rsp+20h] [rbp-50h]
  __int64 v25; // [rsp+30h] [rbp-40h] BYREF
  int v26; // [rsp+38h] [rbp-38h]

  v4 = sub_37946F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = v5;
  v7 = sub_37946F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v8 = *(_QWORD *)(a2 + 80);
  v9 = v7;
  v10 = *(_DWORD *)(a2 + 28);
  v12 = v11;
  v13 = *(_QWORD **)(a1 + 8);
  v14 = *(unsigned __int16 *)(*(_QWORD *)(v4 + 48) + 16LL * (unsigned int)v6);
  v15 = *(_QWORD *)(*(_QWORD *)(v4 + 48) + 16LL * (unsigned int)v6 + 8);
  v25 = v8;
  if ( v8 )
  {
    v21 = v14;
    v22 = v10;
    v23 = v15;
    v24 = v13;
    sub_B96E90((__int64)&v25, v8, 1);
    v14 = v21;
    v10 = v22;
    v15 = v23;
    v13 = v24;
  }
  *((_QWORD *)&v20 + 1) = v12;
  *(_QWORD *)&v20 = v9;
  v16 = *(_DWORD *)(a2 + 24);
  *((_QWORD *)&v19 + 1) = v6;
  *(_QWORD *)&v19 = v4;
  v26 = *(_DWORD *)(a2 + 72);
  v17 = sub_3405C90(v13, v16, (__int64)&v25, v14, v15, v10, a3, v19, v20);
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  return v17;
}
