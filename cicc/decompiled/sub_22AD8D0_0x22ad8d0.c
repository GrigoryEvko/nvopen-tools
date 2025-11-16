// Function: sub_22AD8D0
// Address: 0x22ad8d0
//
_DWORD *__fastcall sub_22AD8D0(_DWORD *a1, __int64 a2, unsigned int *a3)
{
  __int64 v3; // rsi
  _DWORD *v4; // r8
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // rdx
  unsigned int *v8; // rcx

  v3 = a2 - (_QWORD)a1;
  v4 = a1;
  v5 = 0x8E38E38E38E38E39LL * (v3 >> 3);
  if ( v3 > 0 )
  {
    v6 = *a3;
    do
    {
      while ( 1 )
      {
        v7 = v5 >> 1;
        v8 = &v4[18 * (v5 >> 1)];
        if ( v6 > *v8 )
          break;
        v4 = v8 + 18;
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
