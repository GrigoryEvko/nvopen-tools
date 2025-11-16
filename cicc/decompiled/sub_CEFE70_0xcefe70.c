// Function: sub_CEFE70
// Address: 0xcefe70
//
__int64 __fastcall sub_CEFE70(__int64 a1, __int64 a2)
{
  unsigned int i; // r13d
  __int16 v4; // ax
  int v5; // eax
  __int64 v6; // rdx
  int v7; // r14d
  _QWORD *v8; // rax
  _QWORD *v9; // rbx
  __int64 v10; // r15

  for ( i = 0; ; i += v5 )
  {
    v4 = *(_WORD *)(a2 + 24);
    if ( (unsigned __int16)(v4 - 2) <= 2u || v4 == 14 )
    {
      do
      {
        do
        {
          a2 = *(_QWORD *)(a2 + 32);
          ++i;
          v4 = *(_WORD *)(a2 + 24);
        }
        while ( (unsigned __int16)(v4 - 2) <= 2u );
      }
      while ( v4 == 14 );
    }
    if ( v4 != 7 )
      break;
    v5 = sub_CEFE70(a1, *(_QWORD *)(a2 + 32));
    a2 = *(_QWORD *)(a2 + 40);
  }
  if ( (unsigned __int16)(v4 - 8) > 5u && (unsigned __int16)(v4 - 5) > 1u )
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
        v7 += sub_CEFE70(a1, *v8);
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
