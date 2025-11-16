// Function: sub_391A6C0
// Address: 0x391a6c0
//
_BYTE *__fastcall sub_391A6C0(unsigned __int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // r13d
  char v7; // si
  _BYTE *result; // rax
  unsigned int v9; // r14d
  _BYTE *v10; // rax

  v4 = 0;
  do
  {
    while ( 1 )
    {
      ++v4;
      v7 = a1 & 0x7F;
      a1 >>= 7;
      if ( a1 || v4 < a3 )
        v7 |= 0x80u;
      result = *(_BYTE **)(a2 + 24);
      if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 16) )
        break;
      *(_QWORD *)(a2 + 24) = result + 1;
      *result = v7;
      if ( !a1 )
        goto LABEL_7;
    }
    result = (_BYTE *)sub_16E7DE0(a2, v7);
  }
  while ( a1 );
LABEL_7:
  if ( v4 < a3 )
  {
    v9 = a3 - 1;
    if ( v4 < v9 )
    {
      do
      {
        v10 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v10 < *(_QWORD *)(a2 + 16) )
        {
          *(_QWORD *)(a2 + 24) = v10 + 1;
          *v10 = 0x80;
        }
        else
        {
          sub_16E7DE0(a2, 128);
        }
        ++v4;
      }
      while ( v9 != v4 );
    }
    result = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 16) )
    {
      return (_BYTE *)sub_16E7DE0(a2, 0);
    }
    else
    {
      *(_QWORD *)(a2 + 24) = result + 1;
      *result = 0;
    }
  }
  return result;
}
