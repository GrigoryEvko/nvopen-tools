// Function: sub_1CCB2B0
// Address: 0x1ccb2b0
//
__int64 __fastcall sub_1CCB2B0(__int64 a1, __int64 a2)
{
  unsigned int i; // r13d
  int j; // eax
  int v5; // eax
  __int64 v6; // rdx
  int v7; // r14d
  _QWORD *v8; // rax
  _QWORD *v9; // rbx
  __int64 v10; // r15

  for ( i = 0; ; i += v5 )
  {
    for ( j = *(unsigned __int16 *)(a2 + 24); (unsigned __int16)(j - 1) <= 2u; j = *(unsigned __int16 *)(a2 + 24) )
    {
      a2 = *(_QWORD *)(a2 + 32);
      ++i;
    }
    if ( (_WORD)j != 6 )
      break;
    v5 = sub_1CCB2B0(a1, *(_QWORD *)(a2 + 32));
    a2 = *(_QWORD *)(a2 + 40);
  }
  if ( (unsigned __int16)(j - 7) > 2u && (unsigned int)(j - 4) > 1 )
  {
    ++i;
  }
  else
  {
    v6 = *(_QWORD *)(a2 + 40);
    v7 = v6;
    if ( (_DWORD)v6 )
    {
      v8 = *(_QWORD **)(a2 + 32);
      v9 = v8 + 1;
      v10 = (__int64)&v8[(unsigned int)(v6 - 1) + 1];
      while ( 1 )
      {
        v7 += sub_1CCB2B0(a1, *v8);
        v8 = v9;
        if ( (_QWORD *)v10 == v9 )
          break;
        ++v9;
      }
      i += v7;
    }
  }
  return i;
}
