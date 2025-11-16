// Function: sub_30C8EE0
// Address: 0x30c8ee0
//
_QWORD *__fastcall sub_30C8EE0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v6; // rbx
  _QWORD *v8; // rsi
  __int64 v9; // rcx
  unsigned int v10; // edx

  v4 = a2 + 72;
  v6 = *(_QWORD *)(a2 + 80);
  memset(a1, 0, 0x160u);
  if ( v6 != a2 + 72 )
  {
    do
    {
      if ( v6 )
      {
        v8 = (_QWORD *)(v6 - 24);
        v9 = (unsigned int)(*(_DWORD *)(v6 + 20) + 1);
        v10 = *(_DWORD *)(v6 + 20) + 1;
      }
      else
      {
        v8 = 0;
        v9 = 0;
        v10 = 0;
      }
      if ( v10 < *(_DWORD *)(a3 + 32) && *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v9) )
        sub_30C4100(a1, v8);
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v4 != v6 );
  }
  sub_30C8C60(a1, a2, a4);
  return a1;
}
