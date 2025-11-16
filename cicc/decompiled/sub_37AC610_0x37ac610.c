// Function: sub_37AC610
// Address: 0x37ac610
//
unsigned __int8 *__fastcall sub_37AC610(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // r14
  unsigned int v5; // r13d
  __int128 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rsi
  _QWORD *v9; // r8
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rsi
  _QWORD *v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned int v16; // esi
  unsigned __int16 *v17; // rax
  __int128 v18; // rax
  _QWORD *v19; // r15
  __int128 v20; // rax
  unsigned __int8 *v21; // r14
  _QWORD *v23; // [rsp+0h] [rbp-70h]
  __int128 v24; // [rsp+0h] [rbp-70h]
  _QWORD *v25; // [rsp+10h] [rbp-60h]
  __int128 v26; // [rsp+10h] [rbp-60h]
  __int128 v27; // [rsp+20h] [rbp-50h]
  __int128 v28; // [rsp+20h] [rbp-50h]
  __int64 v29; // [rsp+30h] [rbp-40h] BYREF
  int v30; // [rsp+38h] [rbp-38h]

  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v5 = **(unsigned __int16 **)(a2 + 48);
  *(_QWORD *)&v6 = sub_379AB60(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *(_QWORD **)(a1 + 8);
  v27 = v6;
  v29 = v8;
  if ( v8 )
  {
    v25 = v9;
    sub_B96E90((__int64)&v29, v8, 1);
    v9 = v25;
  }
  v30 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v26 = sub_34104F0(v9, *(_QWORD *)(a2 + 40) + 40LL, (__int64)&v29, a3, v7, (__int64)v9);
  *((_QWORD *)&v26 + 1) = v11;
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  v12 = *(_QWORD *)(a2 + 80);
  v13 = *(_QWORD **)(a1 + 8);
  v29 = v12;
  if ( v12 )
  {
    v23 = v13;
    sub_B96E90((__int64)&v29, v12, 1);
    v13 = v23;
  }
  v30 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v24 = sub_34104F0(v13, *(_QWORD *)(a2 + 40) + 80LL, (__int64)&v29, a3, v10, (__int64)v13);
  *((_QWORD *)&v24 + 1) = v14;
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  v15 = *(_QWORD *)(a2 + 80);
  v29 = v15;
  if ( v15 )
    sub_B96E90((__int64)&v29, v15, 1);
  v16 = *(_DWORD *)(a2 + 24);
  v30 = *(_DWORD *)(a2 + 72);
  v17 = (unsigned __int16 *)(*(_QWORD *)(v26 + 48) + 16LL * DWORD2(v26));
  *(_QWORD *)&v18 = sub_340F900(
                      *(_QWORD **)(a1 + 8),
                      v16,
                      (__int64)&v29,
                      *v17,
                      *((_QWORD *)v17 + 1),
                      v26,
                      v27,
                      v26,
                      v24);
  v19 = *(_QWORD **)(a1 + 8);
  v28 = v18;
  *(_QWORD *)&v20 = sub_3400EE0((__int64)v19, 0, (__int64)&v29, 0, a3);
  v21 = sub_3406EB0(v19, 0xA1u, (__int64)&v29, v5, v4, *((__int64 *)&v28 + 1), v28, v20);
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  return v21;
}
