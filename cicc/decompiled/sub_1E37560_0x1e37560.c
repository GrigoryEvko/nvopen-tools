// Function: sub_1E37560
// Address: 0x1e37560
//
_DWORD **__fastcall sub_1E37560(_DWORD **a1, __int64 a2, unsigned int **a3)
{
  __int64 v3; // rsi
  _DWORD **v4; // r8
  __int64 v5; // rax
  unsigned int v6; // edi
  __int64 v7; // rcx
  unsigned int **v8; // rdx

  v3 = a2 - (_QWORD)a1;
  v4 = a1;
  v5 = v3 >> 4;
  if ( v3 > 0 )
  {
    v6 = **a3;
    do
    {
      while ( 1 )
      {
        v7 = v5 >> 1;
        v8 = &v4[2 * (v5 >> 1)];
        if ( **v8 <= v6 )
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
