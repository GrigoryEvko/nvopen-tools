// Function: sub_1BF1750
// Address: 0x1bf1750
//
__int64 __fastcall sub_1BF1750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v13; // [rsp+18h] [rbp-58h] BYREF
  __m128i v14[5]; // [rsp+20h] [rbp-50h] BYREF

  v8 = **(_QWORD **)(a5 + 32);
  sub_13FD840(&v13, a5);
  if ( a6 )
  {
    v9 = *(_QWORD *)(a6 + 48);
    v8 = *(_QWORD *)(a6 + 40);
    if ( v9 )
    {
      if ( v13 )
      {
        sub_161E7C0((__int64)&v13, v13);
        v9 = *(_QWORD *)(a6 + 48);
        v13 = v9;
        if ( !v9 )
          goto LABEL_5;
      }
      else
      {
        v13 = *(_QWORD *)(a6 + 48);
      }
      sub_1623A60((__int64)&v13, v9, 2);
    }
  }
LABEL_5:
  sub_15C9090((__int64)v14, &v13);
  sub_15CA680(a1, a2, a3, a4, v14, v8);
  sub_15CAB20(a1, "loop not vectorized: ", 0x15u);
  if ( v13 )
    sub_161E7C0((__int64)&v13, v13);
  return a1;
}
