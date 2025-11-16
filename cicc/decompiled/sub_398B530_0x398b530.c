// Function: sub_398B530
// Address: 0x398b530
//
void __fastcall sub_398B530(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  _BYTE **i; // rbx
  _BYTE *v7; // rsi

  if ( a2 )
  {
    v5 = 8LL * *(unsigned int *)(a2 + 8);
    for ( i = (_BYTE **)(a2 - v5); (_BYTE **)a2 != i; ++i )
    {
      while ( 1 )
      {
        v7 = *i;
        if ( **i != 29 )
          break;
        ++i;
        sub_398B390(a1, (__int64)v7);
        if ( (_BYTE **)a2 == i )
          return;
      }
      sub_398B4A0(a1, (__int64)v7, a3);
    }
  }
}
