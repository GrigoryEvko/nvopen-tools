// Function: sub_22EC090
// Address: 0x22ec090
//
__int64 __fastcall sub_22EC090(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 result; // rax
  __int64 v9; // rbx
  __int64 v14; // rsi
  __int64 v15; // [rsp-10h] [rbp-50h]
  __int64 i; // [rsp+8h] [rbp-38h]

  result = a2 + 24;
  v9 = *(_QWORD *)(a2 + 32);
  for ( i = a2 + 24; i != v9; v9 = *(_QWORD *)(v9 + 8) )
  {
    v14 = v9 - 56;
    if ( !v9 )
      v14 = 0;
    sub_22EBD50(a1, v14, a3, a4, a5, a6, a7, a8);
    result = v15;
  }
  return result;
}
