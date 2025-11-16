// Function: sub_3798780
// Address: 0x3798780
//
unsigned __int8 *__fastcall sub_3798780(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // r10
  unsigned __int16 *v7; // rax
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // r9
  __int64 v15; // r8
  unsigned int v16; // ebx
  unsigned __int8 *v17; // r12
  bool v19; // al
  __int64 v20; // rcx
  __int16 v21; // ax
  __int64 v22; // [rsp+0h] [rbp-70h]
  __int64 v23; // [rsp+8h] [rbp-68h]
  __int64 v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+10h] [rbp-60h]
  __int64 v26; // [rsp+10h] [rbp-60h]
  unsigned int v27; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+18h] [rbp-58h]
  __int64 v29; // [rsp+18h] [rbp-58h]
  __int64 v30; // [rsp+20h] [rbp-50h] BYREF
  int v31; // [rsp+28h] [rbp-48h]
  __int64 v32; // [rsp+30h] [rbp-40h] BYREF
  __int64 v33; // [rsp+38h] [rbp-38h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *v4;
  sub_37946F0(a1, *v4, v4[1]);
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(unsigned __int16 **)(a2 + 48);
  v8 = *((_QWORD *)v7 + 1);
  LODWORD(v9) = *v7;
  v33 = v8;
  LOWORD(v32) = v9;
  if ( (_WORD)v9 )
  {
    if ( (unsigned __int16)(v9 - 17) <= 0xD3u )
    {
      v8 = 0;
      LOWORD(v9) = word_4456580[(int)v9 - 1];
    }
  }
  else
  {
    v24 = v8;
    v27 = v9;
    v29 = v6;
    v19 = sub_30070B0((__int64)&v32);
    v6 = v29;
    LOWORD(v9) = v27;
    v8 = v24;
    if ( v19 )
    {
      v21 = sub_3009970((__int64)&v32, v5, v27, v20, v24);
      v6 = v29;
      v8 = v9;
      LOWORD(v9) = v21;
    }
  }
  v10 = *(_QWORD *)(a2 + 80);
  v11 = (unsigned __int16)v9;
  v30 = v10;
  if ( v10 )
  {
    v22 = (unsigned __int16)v9;
    v23 = v8;
    v25 = v6;
    sub_B96E90((__int64)&v30, v10, 1);
    v11 = v22;
    v8 = v23;
    v6 = v25;
  }
  v12 = *(unsigned int *)(a2 + 24);
  v31 = *(_DWORD *)(a2 + 72);
  sub_33FAF80(v6, v12, (__int64)&v30, v11, v8, (unsigned int)&v30, a3);
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  v13 = *(_QWORD *)(a2 + 80);
  v14 = *(_QWORD *)(a1 + 8);
  v15 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v16 = **(unsigned __int16 **)(a2 + 48);
  v32 = v13;
  if ( v13 )
  {
    v26 = v15;
    v28 = v14;
    sub_B96E90((__int64)&v32, v13, 1);
    v15 = v26;
    v14 = v28;
  }
  LODWORD(v33) = *(_DWORD *)(a2 + 72);
  v17 = sub_33FAF80(v14, 167, (__int64)&v32, v16, v15, v14, a3);
  if ( v32 )
    sub_B91220((__int64)&v32, v32);
  return v17;
}
