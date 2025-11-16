// Function: sub_379A210
// Address: 0x379a210
//
unsigned __int8 *__fastcall sub_379A210(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 *v4; // rax
  __int64 v5; // r12
  __int64 v6; // r13
  unsigned int v7; // r15d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  int v11; // r9d
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // rcx
  __int64 v15; // r8
  unsigned __int8 *v16; // r12
  __int128 v18; // [rsp-20h] [rbp-90h]
  __int128 v19; // [rsp-10h] [rbp-80h]
  __int64 v20; // [rsp+0h] [rbp-70h]
  __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 v22; // [rsp+10h] [rbp-60h]
  int v23; // [rsp+1Ch] [rbp-54h]
  unsigned __int64 v24; // [rsp+20h] [rbp-50h]
  __int64 v25; // [rsp+20h] [rbp-50h]
  __int64 v26; // [rsp+28h] [rbp-48h]
  _QWORD *v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h] BYREF
  int v29; // [rsp+38h] [rbp-38h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *v4;
  v6 = v4[1];
  v24 = v4[5];
  v26 = v4[6];
  v7 = sub_33CB000(*(_DWORD *)(a2 + 24));
  v8 = sub_37946F0(a1, v24, v26);
  v10 = *(_QWORD *)(a2 + 80);
  v11 = *(_DWORD *)(a2 + 28);
  v12 = v8;
  v13 = v9;
  v27 = *(_QWORD **)(a1 + 8);
  v14 = **(unsigned __int16 **)(a2 + 48);
  v15 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v28 = v10;
  if ( v10 )
  {
    v21 = v9;
    v22 = v14;
    v20 = v8;
    v23 = v11;
    v25 = v15;
    sub_B96E90((__int64)&v28, v10, 1);
    v14 = v22;
    v12 = v20;
    v13 = v21;
    v11 = v23;
    v15 = v25;
  }
  *((_QWORD *)&v19 + 1) = v13;
  *(_QWORD *)&v19 = v12;
  *((_QWORD *)&v18 + 1) = v6;
  *(_QWORD *)&v18 = v5;
  v29 = *(_DWORD *)(a2 + 72);
  v16 = sub_3405C90(v27, v7, (__int64)&v28, v14, v15, v11, a3, v18, v19);
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
  return v16;
}
