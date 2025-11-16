// Function: sub_3776F10
// Address: 0x3776f10
//
unsigned __int8 *__fastcall sub_3776F10(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r9
  __int64 v8; // rax
  unsigned __int16 *v9; // rdx
  unsigned __int8 *v10; // r14
  __int64 v11; // rsi
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // r8
  __int16 v15; // di
  __int64 v16; // rax
  __int64 v18; // rsi
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+10h] [rbp-40h] BYREF
  __int64 v25; // [rsp+18h] [rbp-38h]

  v6 = a1;
  v8 = *(_QWORD *)(a2 + 40);
  v9 = *(unsigned __int16 **)(a2 + 48);
  v10 = *(unsigned __int8 **)(v8 + 40);
  v11 = *(unsigned int *)(v8 + 48);
  v12 = *v9;
  v13 = *((_QWORD *)v9 + 1);
  LOWORD(v24) = v12;
  v25 = v13;
  if ( (_WORD)v12 )
  {
    v14 = 0;
    v15 = word_4456580[v12 - 1];
  }
  else
  {
    v20 = sub_3009970((__int64)&v24, v11, v13, a5, a6);
    v6 = a1;
    a5 = v20;
    v15 = v20;
    v14 = v21;
  }
  LOWORD(a5) = v15;
  v16 = *((_QWORD *)v10 + 6) + 16LL * (unsigned int)v11;
  if ( *(_WORD *)v16 != v15 || *(_QWORD *)(v16 + 8) != v14 && !v15 )
  {
    v18 = *(_QWORD *)(a2 + 80);
    v19 = *(_QWORD *)(v6 + 8);
    v24 = v18;
    if ( v18 )
    {
      v22 = a5;
      v23 = v14;
      sub_B96E90((__int64)&v24, v18, 1);
      a5 = v22;
      v14 = v23;
    }
    LODWORD(v25) = *(_DWORD *)(a2 + 72);
    v10 = sub_33FAF80(v19, 216, (__int64)&v24, a5, v14, v6, a3);
    if ( v24 )
      sub_B91220((__int64)&v24, v24);
  }
  return v10;
}
