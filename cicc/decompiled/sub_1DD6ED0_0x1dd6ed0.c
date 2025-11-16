// Function: sub_1DD6ED0
// Address: 0x1dd6ed0
//
__int64 *__fastcall sub_1DD6ED0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v5; // rsi

  v3 = a2 + 24;
  if ( a3 == v3 )
    goto LABEL_4;
  while ( (unsigned __int16)(**(_WORD **)(a3 + 16) - 12) <= 1u )
  {
    a3 = *(_QWORD *)(a3 + 8);
    if ( a3 == v3 )
      goto LABEL_4;
  }
  if ( v3 == a3 )
  {
LABEL_4:
    *a1 = 0;
    return a1;
  }
  v5 = *(_QWORD *)(a3 + 64);
  *a1 = v5;
  if ( !v5 )
    return a1;
  sub_1623A60((__int64)a1, v5, 2);
  return a1;
}
