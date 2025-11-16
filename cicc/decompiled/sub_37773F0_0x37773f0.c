// Function: sub_37773F0
// Address: 0x37773f0
//
_QWORD *__fastcall sub_37773F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 *v5; // rdx
  _QWORD *v6; // r12
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // r8
  unsigned __int16 v10; // ax
  _QWORD *v11; // r12
  __int64 v13; // rdx
  __int16 v14; // [rsp+0h] [rbp-40h] BYREF
  __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+10h] [rbp-30h] BYREF
  int v17; // [rsp+18h] [rbp-28h]

  v5 = *(unsigned __int16 **)(a2 + 48);
  v6 = *(_QWORD **)(a1 + 8);
  v7 = *v5;
  v8 = *((_QWORD *)v5 + 1);
  v14 = v7;
  v15 = v8;
  if ( (_WORD)v7 )
  {
    v9 = 0;
    v10 = word_4456580[v7 - 1];
  }
  else
  {
    v10 = sub_3009970((__int64)&v14, a2, v8, a4, a5);
    v9 = v13;
  }
  v16 = 0;
  v17 = 0;
  v11 = sub_33F17F0(v6, 51, (__int64)&v16, v10, v9);
  if ( v16 )
    sub_B91220((__int64)&v16, v16);
  return v11;
}
