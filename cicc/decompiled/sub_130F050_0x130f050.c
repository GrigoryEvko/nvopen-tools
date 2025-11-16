// Function: sub_130F050
// Address: 0x130f050
//
_BYTE *__fastcall sub_130F050(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rbx
  __int64 v4; // rsi
  _BYTE *result; // rax

  if ( *(_QWORD *)(a2 + 64) )
  {
    v3 = 0;
    do
    {
      v4 = 9 * v3++;
      result = sub_130B060(a1, (__int64 *)(*(_QWORD *)(a2 + 104) + 16 * v4));
    }
    while ( *(_QWORD *)(a2 + 64) > v3 );
  }
  return result;
}
