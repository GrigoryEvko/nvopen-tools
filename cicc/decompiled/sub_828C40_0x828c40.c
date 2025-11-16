// Function: sub_828C40
// Address: 0x828c40
//
_QWORD *__fastcall sub_828C40(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 i; // rsi

  result = a1;
  if ( !a2 || *(_BYTE *)(a2 + 17) != 3 )
  {
    if ( *((_BYTE *)a1 + 140) == 12 )
    {
      do
        result = (_QWORD *)result[20];
      while ( *((_BYTE *)result + 140) == 12 );
    }
    if ( *(_QWORD *)(result[21] + 40LL) )
    {
      for ( i = *(_QWORD *)(a1[21] + 40LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      return sub_73F0A0((__m128i *)a1, i);
    }
    else
    {
      return (_QWORD *)sub_72D2E0(a1);
    }
  }
  return result;
}
