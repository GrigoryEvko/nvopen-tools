// Function: sub_16CD420
// Address: 0x16cd420
//
_BYTE *__fastcall sub_16CD420(__int64 a1, unsigned __int8 *a2, int a3)
{
  unsigned __int8 *v3; // r14
  int v4; // ebx
  __int64 v5; // r13
  _BYTE *v6; // rax
  __int64 v7; // rsi
  _BYTE *v8; // rax
  _BYTE *result; // rax

  if ( a3 )
  {
    v3 = a2;
    v4 = 0;
    v5 = (__int64)&a2[a3 - 1 + 1];
    do
    {
      v7 = *v3;
      if ( (_BYTE)v7 == 9 )
      {
        do
        {
          v8 = *(_BYTE **)(a1 + 24);
          if ( (unsigned __int64)v8 < *(_QWORD *)(a1 + 16) )
          {
            *(_QWORD *)(a1 + 24) = v8 + 1;
            *v8 = 32;
          }
          else
          {
            sub_16E7DE0(a1, 32);
          }
          ++v4;
        }
        while ( (v4 & 7) != 0 );
      }
      else
      {
        v6 = *(_BYTE **)(a1 + 24);
        if ( *(_QWORD *)(a1 + 16) <= (unsigned __int64)v6 )
        {
          sub_16E7DE0(a1, v7);
        }
        else
        {
          *(_QWORD *)(a1 + 24) = v6 + 1;
          *v6 = v7;
        }
        ++v4;
      }
      ++v3;
    }
    while ( v3 != (unsigned __int8 *)v5 );
  }
  result = *(_BYTE **)(a1 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(a1 + 16) )
    return (_BYTE *)sub_16E7DE0(a1, 10);
  *(_QWORD *)(a1 + 24) = result + 1;
  *result = 10;
  return result;
}
