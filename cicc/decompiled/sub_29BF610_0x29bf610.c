// Function: sub_29BF610
// Address: 0x29bf610
//
_QWORD *__fastcall sub_29BF610(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 *a4)
{
  __int64 v4; // rsi
  _QWORD *v5; // r8
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned int v8; // r9d
  __int64 v9; // rcx
  _QWORD *v10; // rsi

  v4 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = v4 >> 3;
  if ( v4 > 0 )
  {
    v7 = *a4;
    v8 = *(_DWORD *)(*a4 + 16LL * *a3);
    do
    {
      while ( 1 )
      {
        v9 = v6 >> 1;
        v10 = &v5[v6 >> 1];
        if ( v8 < *(_DWORD *)(v7 + 16LL * *v10) )
          break;
        v5 = v10 + 1;
        v6 = v6 - v9 - 1;
        if ( v6 <= 0 )
          return v5;
      }
      v6 >>= 1;
    }
    while ( v9 > 0 );
  }
  return v5;
}
