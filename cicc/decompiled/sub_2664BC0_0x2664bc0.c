// Function: sub_2664BC0
// Address: 0x2664bc0
//
_QWORD *__fastcall sub_2664BC0(_QWORD *a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 v3; // rsi
  _QWORD *v4; // r8
  __int64 v5; // rax
  unsigned __int64 v6; // rsi
  __int64 v7; // rcx
  unsigned __int64 *v8; // rdx

  v3 = a2 - (_QWORD)a1;
  v4 = a1;
  v5 = v3 >> 4;
  if ( v3 > 0 )
  {
    v6 = *a3;
    do
    {
      while ( 1 )
      {
        v7 = v5 >> 1;
        v8 = &v4[2 * (v5 >> 1)];
        if ( *v8 >= v6 )
          break;
        v4 = v8 + 2;
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
