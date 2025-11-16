// Function: sub_3776E20
// Address: 0x3776e20
//
unsigned __int8 *__fastcall sub_3776E20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  unsigned __int16 *v8; // rdx
  _QWORD *v9; // r12
  __int128 *v10; // r15
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rsi
  unsigned __int8 *v15; // r12
  __int64 v17; // rdx
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h] BYREF
  int v20; // [rsp+18h] [rbp-48h]
  __int16 v21; // [rsp+20h] [rbp-40h] BYREF
  __int64 v22; // [rsp+28h] [rbp-38h]

  v8 = *(unsigned __int16 **)(a2 + 48);
  v9 = *(_QWORD **)(a1 + 8);
  v10 = *(__int128 **)(a2 + 40);
  v11 = *v8;
  v12 = *((_QWORD *)v8 + 1);
  v21 = v11;
  v22 = v12;
  if ( (_WORD)v11 )
  {
    v13 = 0;
    LOWORD(v11) = word_4456580[v11 - 1];
  }
  else
  {
    v11 = sub_3009970((__int64)&v21, a2, v12, a4, a5);
    HIWORD(v6) = HIWORD(v11);
    v13 = v17;
  }
  v14 = *(_QWORD *)(a2 + 80);
  LOWORD(v6) = v11;
  v19 = v14;
  if ( v14 )
  {
    v18 = v13;
    sub_B96E90((__int64)&v19, v14, 1);
    v13 = v18;
  }
  v20 = *(_DWORD *)(a2 + 72);
  v15 = sub_3406EB0(v9, 0x9Eu, (__int64)&v19, v6, v13, a6, *v10, *(__int128 *)((char *)v10 + 40));
  if ( v19 )
    sub_B91220((__int64)&v19, v19);
  return v15;
}
