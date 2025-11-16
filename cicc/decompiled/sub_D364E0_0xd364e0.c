// Function: sub_D364E0
// Address: 0xd364e0
//
__int64 __fastcall sub_D364E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 result; // rax
  __int64 v10; // rdi
  __int64 v12; // [rsp+8h] [rbp-58h]
  __int64 v13; // [rsp+8h] [rbp-58h]
  __int64 v14; // [rsp+18h] [rbp-48h] BYREF
  __m128i v15[4]; // [rsp+20h] [rbp-40h] BYREF

  v7 = **(_QWORD **)(*(_QWORD *)(a1 + 24) + 32LL);
  sub_D4BD20(&v14);
  if ( a4 )
  {
    v8 = *(_QWORD *)(a4 + 48);
    v7 = *(_QWORD *)(a4 + 40);
    if ( v8 )
    {
      if ( v14 )
      {
        sub_B91220((__int64)&v14, v14);
        v8 = *(_QWORD *)(a4 + 48);
        v14 = v8;
        if ( !v8 )
          goto LABEL_5;
      }
      else
      {
        v14 = *(_QWORD *)(a4 + 48);
      }
      sub_B96E90((__int64)&v14, v8, 1);
    }
  }
LABEL_5:
  sub_B157E0((__int64)v15, &v14);
  result = sub_22077B0(432);
  if ( result )
  {
    v12 = result;
    sub_B17850(result, (__int64)"loop-accesses", a2, a3, v15, v7);
    result = v12;
  }
  v10 = *(_QWORD *)(a1 + 112);
  *(_QWORD *)(a1 + 112) = result;
  if ( v10 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 16LL))(v10);
    result = *(_QWORD *)(a1 + 112);
  }
  if ( v14 )
  {
    v13 = result;
    sub_B91220((__int64)&v14, v14);
    return v13;
  }
  return result;
}
