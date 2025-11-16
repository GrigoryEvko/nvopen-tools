// Function: sub_3798950
// Address: 0x3798950
//
unsigned __int8 *__fastcall sub_3798950(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rax
  __int64 v5; // r9
  _QWORD *v6; // r11
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r13
  unsigned __int16 *v10; // rax
  unsigned int v11; // esi
  __int64 v12; // r8
  __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned int v15; // esi
  __int64 v16; // rsi
  __int64 v17; // r9
  __int64 v18; // r8
  unsigned int v19; // ebx
  unsigned __int8 *v20; // r12
  bool v22; // al
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int16 v25; // ax
  __int64 v26; // rdx
  __int128 v27; // [rsp-20h] [rbp-A0h]
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  _QWORD *v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+20h] [rbp-60h]
  _QWORD *v33; // [rsp+20h] [rbp-60h]
  __int64 v34; // [rsp+28h] [rbp-58h]
  __int64 v35; // [rsp+28h] [rbp-58h]
  __int64 v36; // [rsp+30h] [rbp-50h] BYREF
  int v37; // [rsp+38h] [rbp-48h]
  __int64 v38; // [rsp+40h] [rbp-40h] BYREF
  __int64 v39; // [rsp+48h] [rbp-38h]

  v4 = sub_37946F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(_QWORD **)(a1 + 8);
  v7 = v4;
  v9 = v8;
  v34 = *(_QWORD *)(a2 + 40);
  v10 = *(unsigned __int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOWORD(v38) = v11;
  v39 = v12;
  if ( (_WORD)v11 )
  {
    if ( (unsigned __int16)(v11 - 17) <= 0xD3u )
    {
      v12 = 0;
      LOWORD(v11) = word_4456580[v11 - 1];
    }
  }
  else
  {
    v30 = v12;
    v33 = v6;
    v22 = sub_30070B0((__int64)&v38);
    v6 = v33;
    v12 = v30;
    if ( v22 )
    {
      v25 = sub_3009970((__int64)&v38, v11, v23, v24, v30);
      v6 = v33;
      LOWORD(v11) = v25;
      v12 = v26;
    }
  }
  v13 = (unsigned __int16)v11;
  v14 = *(_QWORD *)(a2 + 80);
  v36 = v14;
  if ( v14 )
  {
    v28 = v13;
    v29 = v12;
    v31 = v6;
    sub_B96E90((__int64)&v36, v14, 1);
    v13 = v28;
    v12 = v29;
    v6 = v31;
  }
  v15 = *(_DWORD *)(a2 + 24);
  v37 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v27 + 1) = v9;
  *(_QWORD *)&v27 = v7;
  sub_3406EB0(v6, v15, (__int64)&v36, v13, v12, v5, v27, *(_OWORD *)(v34 + 40));
  if ( v36 )
    sub_B91220((__int64)&v36, v36);
  v16 = *(_QWORD *)(a2 + 80);
  v17 = *(_QWORD *)(a1 + 8);
  v18 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v19 = **(unsigned __int16 **)(a2 + 48);
  v38 = v16;
  if ( v16 )
  {
    v32 = v17;
    v35 = v18;
    sub_B96E90((__int64)&v38, v16, 1);
    v17 = v32;
    v18 = v35;
  }
  LODWORD(v39) = *(_DWORD *)(a2 + 72);
  v20 = sub_33FAF80(v17, 167, (__int64)&v38, v19, v18, v17, a3);
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
  return v20;
}
