// Function: sub_3860590
// Address: 0x3860590
//
__int64 __fastcall sub_3860590(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  __int64 v8; // r15
  __int64 v9; // rsi
  __int64 result; // rax
  __int64 v11; // rdi
  __int64 v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+18h] [rbp-58h] BYREF
  __m128i v16[5]; // [rsp+20h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(a1 + 24);
  v8 = **(_QWORD **)(v7 + 32);
  sub_13FD840(&v15, v7);
  if ( a4 )
  {
    v9 = *(_QWORD *)(a4 + 48);
    v8 = *(_QWORD *)(a4 + 40);
    if ( v9 )
    {
      if ( v15 )
      {
        sub_161E7C0((__int64)&v15, v15);
        v9 = *(_QWORD *)(a4 + 48);
        v15 = v9;
        if ( !v9 )
          goto LABEL_5;
      }
      else
      {
        v15 = *(_QWORD *)(a4 + 48);
      }
      sub_1623A60((__int64)&v15, v9, 2);
    }
  }
LABEL_5:
  sub_15C9090((__int64)v16, &v15);
  result = sub_22077B0(0x1D8u);
  if ( result )
  {
    v13 = result;
    sub_15CA680(result, (__int64)"loop-accesses", a2, a3, v16, v8);
    result = v13;
  }
  v11 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(a1 + 56) = result;
  if ( v11 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
    result = *(_QWORD *)(a1 + 56);
  }
  if ( v15 )
  {
    v14 = result;
    sub_161E7C0((__int64)&v15, v15);
    return v14;
  }
  return result;
}
