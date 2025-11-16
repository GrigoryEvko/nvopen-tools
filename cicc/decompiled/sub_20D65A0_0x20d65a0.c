// Function: sub_20D65A0
// Address: 0x20d65a0
//
__int64 *__fastcall sub_20D65A0(__int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r12
  __int16 v4; // ax
  char v5; // al
  __int64 v6; // rsi

  v2 = sub_1DD6160(a2);
  if ( v2 == a2 + 24
    || ((v3 = v2, v4 = *(_WORD *)(v2 + 46), (v4 & 4) != 0) || (v4 & 8) == 0
      ? (v5 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v3 + 16) + 8LL) >> 7)
      : (v5 = sub_1E15D00(v3, 0x80u, 1)),
        !v5) )
  {
    *a1 = 0;
  }
  else
  {
    v6 = *(_QWORD *)(v3 + 64);
    *a1 = v6;
    if ( v6 )
    {
      sub_1623A60((__int64)a1, v6, 2);
      return a1;
    }
  }
  return a1;
}
