// Function: sub_37998C0
// Address: 0x37998c0
//
unsigned __int8 *__fastcall sub_37998C0(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r8
  _QWORD *v10; // r10
  __int64 v11; // r11
  __int64 v12; // rdx
  __int64 v13; // r13
  unsigned __int16 *v14; // rdx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // r9
  __int64 v23; // r8
  unsigned int v24; // r14d
  unsigned __int8 *v25; // r12
  __int64 v27; // rdx
  __int128 v28; // [rsp-20h] [rbp-A0h]
  __int64 v29; // [rsp+8h] [rbp-78h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  __int64 v31; // [rsp+18h] [rbp-68h]
  _QWORD *v32; // [rsp+20h] [rbp-60h]
  __int64 v33; // [rsp+20h] [rbp-60h]
  __int64 v34; // [rsp+20h] [rbp-60h]
  __int64 v36; // [rsp+28h] [rbp-58h]
  _QWORD *v37; // [rsp+28h] [rbp-58h]
  __int64 v38; // [rsp+30h] [rbp-50h] BYREF
  int v39; // [rsp+38h] [rbp-48h]
  __int64 v40; // [rsp+40h] [rbp-40h] BYREF
  __int64 v41; // [rsp+48h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 40);
  v7 = *v6;
  v16 = sub_37946F0(a1, *v6, v6[1]);
  v10 = *(_QWORD **)(a1 + 8);
  v11 = *(_QWORD *)(a2 + 40);
  v13 = v12;
  v14 = *(unsigned __int16 **)(a2 + 48);
  v15 = v16;
  LODWORD(v16) = *v14;
  v17 = *((_QWORD *)v14 + 1);
  LOWORD(v40) = v16;
  v41 = v17;
  if ( (_WORD)v16 )
  {
    v18 = a5;
    v19 = 0;
    LOWORD(v16) = word_4456580[(int)v16 - 1];
  }
  else
  {
    v34 = v11;
    v37 = v10;
    v16 = sub_3009970((__int64)&v40, v7, v17, v8, v9);
    v11 = v34;
    v10 = v37;
    v18 = v16;
    v19 = v27;
  }
  v20 = *(_QWORD *)(a2 + 80);
  LOWORD(v18) = v16;
  v38 = v20;
  if ( v20 )
  {
    v29 = v18;
    v30 = v11;
    v31 = v19;
    v32 = v10;
    sub_B96E90((__int64)&v38, v20, 1);
    v18 = v29;
    v11 = v30;
    v19 = v31;
    v10 = v32;
  }
  v39 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v28 + 1) = v13;
  *(_QWORD *)&v28 = v15;
  sub_3406EB0(v10, 0xE6u, (__int64)&v38, v18, v19, (__int64)&v38, v28, *(_OWORD *)(v11 + 40));
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
  v21 = *(_QWORD *)(a2 + 80);
  v22 = *(_QWORD *)(a1 + 8);
  v23 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v24 = **(unsigned __int16 **)(a2 + 48);
  v40 = v21;
  if ( v21 )
  {
    v33 = v22;
    v36 = v23;
    sub_B96E90((__int64)&v40, v21, 1);
    v22 = v33;
    v23 = v36;
  }
  LODWORD(v41) = *(_DWORD *)(a2 + 72);
  v25 = sub_33FAF80(v22, 167, (__int64)&v40, v24, v23, v22, a3);
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
  return v25;
}
