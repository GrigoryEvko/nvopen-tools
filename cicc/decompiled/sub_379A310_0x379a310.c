// Function: sub_379A310
// Address: 0x379a310
//
unsigned __int8 *__fastcall sub_379A310(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  unsigned __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // r11
  unsigned __int16 *v15; // rdx
  __int64 v16; // r10
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rsi
  _QWORD *v22; // r9
  unsigned int v23; // esi
  __int64 v24; // rsi
  __int64 v25; // r9
  __int64 v26; // r8
  unsigned int v27; // ebx
  unsigned __int8 *v28; // r12
  __int64 v30; // rdx
  __int128 v31; // [rsp-20h] [rbp-90h]
  __int128 v32; // [rsp-10h] [rbp-80h]
  __int64 v33; // [rsp+0h] [rbp-70h]
  __int64 v34; // [rsp+8h] [rbp-68h]
  __int64 v35; // [rsp+10h] [rbp-60h]
  __int64 v36; // [rsp+18h] [rbp-58h]
  __int64 v37; // [rsp+18h] [rbp-58h]
  _QWORD *v39; // [rsp+20h] [rbp-50h]
  __int64 v40; // [rsp+20h] [rbp-50h]
  __int64 v41; // [rsp+20h] [rbp-50h]
  __int64 v42; // [rsp+28h] [rbp-48h]
  __int64 v43; // [rsp+30h] [rbp-40h] BYREF
  __int64 v44; // [rsp+38h] [rbp-38h]

  v6 = sub_37946F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v7 = *(_QWORD *)(a2 + 40);
  v9 = v8;
  v10 = *(_QWORD *)(v7 + 40);
  v17 = sub_37946F0(a1, v10, *(_QWORD *)(v7 + 48));
  v14 = v13;
  v15 = *(unsigned __int16 **)(a2 + 48);
  v16 = v17;
  LODWORD(v17) = *v15;
  v18 = *((_QWORD *)v15 + 1);
  LOWORD(v43) = v17;
  v44 = v18;
  if ( (_WORD)v17 )
  {
    v19 = a5;
    v20 = 0;
    LOWORD(v17) = word_4456580[(int)v17 - 1];
  }
  else
  {
    v41 = v16;
    v42 = v14;
    v17 = sub_3009970((__int64)&v43, v10, v18, v11, v12);
    v16 = v41;
    v14 = v42;
    v19 = v17;
    v20 = v30;
  }
  v21 = *(_QWORD *)(a2 + 80);
  v22 = *(_QWORD **)(a1 + 8);
  LOWORD(v19) = v17;
  v43 = v21;
  if ( v21 )
  {
    v35 = v19;
    v33 = v16;
    v34 = v14;
    v36 = v20;
    v39 = v22;
    sub_B96E90((__int64)&v43, v21, 1);
    v19 = v35;
    v16 = v33;
    v14 = v34;
    v20 = v36;
    v22 = v39;
  }
  *((_QWORD *)&v32 + 1) = v14;
  *(_QWORD *)&v32 = v16;
  v23 = *(_DWORD *)(a2 + 24);
  *((_QWORD *)&v31 + 1) = v9;
  *(_QWORD *)&v31 = v6;
  LODWORD(v44) = *(_DWORD *)(a2 + 72);
  sub_3406EB0(v22, v23, (__int64)&v43, v19, v20, (__int64)v22, v31, v32);
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  v24 = *(_QWORD *)(a2 + 80);
  v25 = *(_QWORD *)(a1 + 8);
  v26 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v27 = **(unsigned __int16 **)(a2 + 48);
  v43 = v24;
  if ( v24 )
  {
    v37 = v25;
    v40 = v26;
    sub_B96E90((__int64)&v43, v24, 1);
    v25 = v37;
    v26 = v40;
  }
  LODWORD(v44) = *(_DWORD *)(a2 + 72);
  v28 = sub_33FAF80(v25, 167, (__int64)&v43, v27, v26, v25, a3);
  if ( v43 )
    sub_B91220((__int64)&v43, v43);
  return v28;
}
