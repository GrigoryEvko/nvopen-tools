// Function: sub_3799D00
// Address: 0x3799d00
//
unsigned __int8 *__fastcall sub_3799D00(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r10
  unsigned __int16 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // r9
  __int64 v20; // r8
  unsigned int v21; // r14d
  unsigned __int8 *v22; // r12
  __int64 v24; // rdx
  __int64 v25; // [rsp+0h] [rbp-70h]
  __int64 v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+20h] [rbp-50h] BYREF
  int v33; // [rsp+28h] [rbp-48h]
  __int64 v34; // [rsp+30h] [rbp-40h] BYREF
  __int64 v35; // [rsp+38h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 40);
  v7 = *v6;
  sub_37946F0(a1, *v6, v6[1]);
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(unsigned __int16 **)(a2 + 48);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  LOWORD(v34) = v12;
  v35 = v13;
  if ( (_WORD)v12 )
  {
    v14 = a5;
    v15 = 0;
    LOWORD(v16) = word_4456580[(int)v12 - 1];
  }
  else
  {
    v31 = v10;
    v16 = sub_3009970((__int64)&v34, v7, v12, v8, v9);
    v10 = v31;
    v14 = v16;
    v15 = v24;
  }
  v17 = *(_QWORD *)(a2 + 80);
  LOWORD(v14) = v16;
  v32 = v17;
  if ( v17 )
  {
    v25 = v14;
    v26 = v15;
    v27 = v10;
    sub_B96E90((__int64)&v32, v17, 1);
    v14 = v25;
    v15 = v26;
    v10 = v27;
  }
  v33 = *(_DWORD *)(a2 + 72);
  sub_33FAF80(v10, 233, (__int64)&v32, v14, v15, (unsigned int)&v32, a3);
  if ( v32 )
    sub_B91220((__int64)&v32, v32);
  v18 = *(_QWORD *)(a2 + 80);
  v19 = *(_QWORD *)(a1 + 8);
  v20 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v21 = **(unsigned __int16 **)(a2 + 48);
  v34 = v18;
  if ( v18 )
  {
    v28 = v20;
    v30 = v19;
    sub_B96E90((__int64)&v34, v18, 1);
    v20 = v28;
    v19 = v30;
  }
  LODWORD(v35) = *(_DWORD *)(a2 + 72);
  v22 = sub_33FAF80(v19, 167, (__int64)&v34, v21, v20, v19, a3);
  if ( v34 )
    sub_B91220((__int64)&v34, v34);
  return v22;
}
