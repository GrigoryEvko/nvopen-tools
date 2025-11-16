// Function: sub_1077AC0
// Address: 0x1077ac0
//
_QWORD *__fastcall sub_1077AC0(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rsi
  _QWORD *v4; // r8
  __int64 v5; // rax
  unsigned __int64 v6; // r9
  __int64 v7; // rdx
  _QWORD *v8; // rsi

  v3 = a2 - (_QWORD)a1;
  v4 = a1;
  v5 = 0xCCCCCCCCCCCCCCCDLL * (v3 >> 3);
  if ( v3 > 0 )
  {
    v6 = *(_QWORD *)(a3[4] + 160LL) + *a3;
    do
    {
      while ( 1 )
      {
        v7 = v5 >> 1;
        v8 = &v4[5 * (v5 >> 1)];
        if ( v6 < *(_QWORD *)(v8[4] + 160LL) + *v8 )
          break;
        v4 = v8 + 5;
        v5 = v5 - v7 - 1;
        if ( v5 <= 0 )
          return v4;
      }
      v5 >>= 1;
    }
    while ( v7 > 0 );
  }
  return v4;
}
