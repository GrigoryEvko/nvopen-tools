// Function: sub_107A5C0
// Address: 0x107a5c0
//
__int64 __fastcall sub_107A5C0(unsigned __int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // r13d
  char v7; // si
  char *v8; // rax
  unsigned int v9; // r14d
  _BYTE *v10; // rax
  _BYTE *v12; // rax

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
      v8 = *(char **)(a2 + 32);
      if ( (unsigned __int64)v8 >= *(_QWORD *)(a2 + 24) )
        break;
      *(_QWORD *)(a2 + 32) = v8 + 1;
      *v8 = v7;
      if ( !a1 )
        goto LABEL_7;
    }
    sub_CB5D20(a2, v7);
  }
  while ( a1 );
LABEL_7:
  if ( v4 < a3 )
  {
    v9 = a3 - 1;
    if ( v9 > v4 )
    {
      do
      {
        v12 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v12 < *(_QWORD *)(a2 + 24) )
        {
          *(_QWORD *)(a2 + 32) = v12 + 1;
          *v12 = 0x80;
        }
        else
        {
          sub_CB5D20(a2, 128);
        }
        ++v4;
      }
      while ( v9 != v4 );
    }
    else
    {
      v9 = v4;
    }
    v10 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(a2 + 24) )
    {
      sub_CB5D20(a2, 0);
    }
    else
    {
      *(_QWORD *)(a2 + 32) = v10 + 1;
      *v10 = 0;
    }
    return v9 + 1;
  }
  return v4;
}
