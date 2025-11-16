// Function: sub_1920010
// Address: 0x1920010
//
_DWORD *__fastcall sub_1920010(_DWORD *a1, __int64 a2, unsigned int *a3)
{
  __int64 v5; // rsi
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // rcx
  unsigned int *v9; // rdx

  v5 = a2 - (_QWORD)a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * (v5 >> 3);
  if ( v5 > 0 )
  {
    v7 = *a3;
    do
    {
      while ( 1 )
      {
        v8 = v6 >> 1;
        v9 = &a1[2 * (v6 >> 1) + 2 * (v6 & 0xFFFFFFFFFFFFFFFELL)];
        if ( *v9 < v7 || *v9 == v7 && v9[1] < a3[1] )
          break;
        v6 >>= 1;
        if ( v8 <= 0 )
          return a1;
      }
      a1 = v9 + 6;
      v6 = v6 - v8 - 1;
    }
    while ( v6 > 0 );
  }
  return a1;
}
