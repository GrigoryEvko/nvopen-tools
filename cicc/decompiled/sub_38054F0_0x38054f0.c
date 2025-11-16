// Function: sub_38054F0
// Address: 0x38054f0
//
unsigned __int8 *__fastcall sub_38054F0(__int64 a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r15d
  __int64 *v5; // rax
  __int64 v6; // rsi
  unsigned __int8 *v7; // rax
  __int64 v8; // r8
  _QWORD *v9; // r9
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // r13
  unsigned __int8 *v13; // r12
  unsigned __int16 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r8
  int v18; // eax
  __int64 v19; // rsi
  unsigned __int8 *v20; // r12
  __int64 v22; // rdx
  __int128 v23; // [rsp-20h] [rbp-90h]
  __int64 v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+10h] [rbp-60h]
  __int64 v26; // [rsp+10h] [rbp-60h]
  _QWORD *v27; // [rsp+18h] [rbp-58h]
  _QWORD *v28; // [rsp+18h] [rbp-58h]
  __int64 v29; // [rsp+20h] [rbp-50h] BYREF
  int v30; // [rsp+28h] [rbp-48h]
  __int16 v31; // [rsp+30h] [rbp-40h] BYREF
  __int64 v32; // [rsp+38h] [rbp-38h]

  v5 = *(__int64 **)(a2 + 40);
  v6 = *v5;
  v7 = sub_375A8B0(a1, *v5, v5[1], a3);
  v9 = *(_QWORD **)(a1 + 8);
  v10 = *(_QWORD *)(a2 + 40);
  v12 = v11;
  v13 = v7;
  v14 = (unsigned __int16 *)(*((_QWORD *)v7 + 6) + 16LL * (unsigned int)v11);
  v15 = *v14;
  v16 = *((_QWORD *)v14 + 1);
  v31 = v15;
  v32 = v16;
  if ( (_WORD)v15 )
  {
    v17 = 0;
    LOWORD(v18) = word_4456580[(int)v15 - 1];
  }
  else
  {
    v26 = v10;
    v28 = v9;
    v18 = sub_3009970((__int64)&v31, v6, v15, v10, v8);
    v10 = v26;
    v9 = v28;
    HIWORD(v3) = HIWORD(v18);
    v17 = v22;
  }
  v19 = *(_QWORD *)(a2 + 80);
  LOWORD(v3) = v18;
  v29 = v19;
  if ( v19 )
  {
    v24 = v10;
    v25 = v17;
    v27 = v9;
    sub_B96E90((__int64)&v29, v19, 1);
    v10 = v24;
    v17 = v25;
    v9 = v27;
  }
  v30 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v23 + 1) = v12;
  *(_QWORD *)&v23 = v13;
  v20 = sub_3406EB0(v9, 0x9Eu, (__int64)&v29, v3, v17, (__int64)v9, v23, *(_OWORD *)(v10 + 40));
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  return v20;
}
