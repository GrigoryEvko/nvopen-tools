// Function: sub_2E32810
// Address: 0x2e32810
//
__int64 *__fastcall sub_2E32810(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int16 v4; // ax
  __int64 v6; // rsi

  v3 = a2 + 48;
  if ( v3 == a3 )
  {
LABEL_5:
    *a1 = 0;
    return a1;
  }
  while ( 1 )
  {
    v4 = *(_WORD *)(a3 + 68);
    if ( (unsigned __int16)(v4 - 14) > 4u && v4 != 24 )
      break;
    a3 = *(_QWORD *)(a3 + 8);
    if ( v3 == a3 )
      goto LABEL_5;
  }
  v6 = *(_QWORD *)(a3 + 56);
  *a1 = v6;
  if ( !v6 )
    return a1;
  sub_B96E90((__int64)a1, v6, 1);
  return a1;
}
