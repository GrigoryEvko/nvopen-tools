// Function: sub_3776CF0
// Address: 0x3776cf0
//
unsigned __int8 *__fastcall sub_3776CF0(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, int a7)
{
  unsigned __int16 *v8; // rdx
  int v9; // eax
  __int64 v10; // rdx
  __int16 v11; // dx
  __int64 v12; // rax
  unsigned __int8 **v13; // rax
  unsigned __int8 *v14; // r14
  __int64 v15; // rsi
  __int64 v16; // rbx
  __int16 v18; // ax
  __int64 v19; // rdx
  __int64 v20; // r8
  unsigned int v21; // [rsp+0h] [rbp-50h] BYREF
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h] BYREF
  __int64 v24; // [rsp+18h] [rbp-38h]

  v8 = *(unsigned __int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v23) = v9;
  v24 = v10;
  if ( (_WORD)v9 )
  {
    v11 = word_4456580[v9 - 1];
    v12 = 0;
  }
  else
  {
    v18 = sub_3009970((__int64)&v23, a2, v10, a5, a6);
    v20 = v19;
    v11 = v18;
    v12 = v20;
  }
  v22 = v12;
  v13 = *(unsigned __int8 ***)(a2 + 40);
  LOWORD(v21) = v11;
  v14 = *v13;
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 2) > 7u
      && (unsigned __int16)(v11 - 17) > 0x6Cu
      && (unsigned __int16)(v11 - 176) > 0x1Fu )
    {
      return v14;
    }
  }
  else if ( !sub_3007070((__int64)&v21) )
  {
    return v14;
  }
  v15 = *(_QWORD *)(a2 + 80);
  v16 = *(_QWORD *)(a1 + 8);
  v23 = v15;
  if ( v15 )
    sub_B96E90((__int64)&v23, v15, 1);
  LODWORD(v24) = *(_DWORD *)(a2 + 72);
  v14 = sub_33FAF80(v16, 216, (__int64)&v23, v21, v22, a7, a3);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v14;
}
