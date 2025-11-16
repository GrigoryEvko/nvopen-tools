// Function: sub_208C170
// Address: 0x208c170
//
__int64 *__fastcall sub_208C170(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 *v5; // rax
  __int64 v6; // r13
  __int64 *v7; // rbx
  __int16 v8; // ax
  __int64 v9; // rsi
  unsigned __int8 *v10; // rsi
  __int64 *v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 *v15; // rax
  __int64 v16; // rsi
  __int64 v17[2]; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v18[7]; // [rsp+28h] [rbp-38h] BYREF

  v17[0] = a2;
  v5 = sub_205F5C0(a1 + 8, v17);
  v6 = v5[1];
  if ( v6 )
  {
    v7 = v5;
    v8 = *(_WORD *)(v6 + 24);
    if ( (unsigned __int16)(v8 - 10) <= 1u || (unsigned __int16)(v8 - 32) <= 1u )
    {
      v18[0] = 0;
      if ( (_QWORD *)(v6 + 72) != v18 )
      {
        v9 = *(_QWORD *)(v6 + 72);
        if ( v9 )
        {
          sub_161E7C0(v6 + 72, v9);
          v10 = (unsigned __int8 *)v18[0];
          *(_QWORD *)(v6 + 72) = v18[0];
          if ( v10 )
            sub_1623210((__int64)v18, v10, v6 + 72);
        }
      }
    }
    return (__int64 *)v7[1];
  }
  else
  {
    v12 = sub_2067630(a1, v17[0], a3, a4, a5);
    v14 = v13;
    v15 = sub_205F5C0(a1 + 8, v17);
    v16 = v17[0];
    v15[1] = (__int64)v12;
    *((_DWORD *)v15 + 4) = v14;
    sub_20540C0(a1, v16, (__int64)v12, v14);
    return v12;
  }
}
