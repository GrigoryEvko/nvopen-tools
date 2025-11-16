// Function: sub_2123C50
// Address: 0x2123c50
//
__int64 __fastcall sub_2123C50(__m128i **a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // rsi
  char v10; // di
  __int64 v11; // rax
  char v12; // dl
  unsigned __int8 v13; // r15
  int v14; // r13d
  unsigned __int64 v15; // rax
  __int64 v16; // rsi
  __m128i *v17; // r10
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // r11
  __int64 v21; // r12
  __int64 v23; // [rsp+0h] [rbp-80h]
  __int64 v24; // [rsp+8h] [rbp-78h]
  __m128i *v25; // [rsp+8h] [rbp-78h]
  _QWORD v26[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v27; // [rsp+20h] [rbp-60h] BYREF
  int v28; // [rsp+28h] [rbp-58h]
  __int64 v29; // [rsp+30h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL);
  v8 = *(_WORD *)(a2 + 24) == 161;
  v9 = *(_QWORD *)(v7 + 8);
  v10 = *(_BYTE *)v7;
  v11 = *(_QWORD *)(a2 + 40);
  v12 = *(_BYTE *)v11;
  v13 = *(_BYTE *)v11;
  if ( v8 )
    v12 = 8;
  v24 = *(_QWORD *)(v11 + 8);
  v14 = sub_1F3FF10(v10, v9, v12);
  v15 = sub_2120330((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v16 = *(_QWORD *)(a2 + 72);
  v17 = *a1;
  v26[0] = v15;
  v18 = v24;
  v26[1] = v19;
  v27 = v16;
  if ( v16 )
  {
    v23 = v24;
    v25 = v17;
    sub_1623A60((__int64)&v27, v16, 2);
    v18 = v23;
    v17 = v25;
  }
  v20 = (__int64)a1[1];
  v28 = *(_DWORD *)(a2 + 64);
  sub_20BE530((__int64)&v29, v17, v20, v14, v13, v18, a3, a4, a5, (__int64)v26, 1u, 0, (__int64)&v27, 0, 1);
  v21 = v29;
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v21;
}
