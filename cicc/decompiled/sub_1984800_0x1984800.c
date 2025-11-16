// Function: sub_1984800
// Address: 0x1984800
//
__int64 *__fastcall sub_1984800(unsigned int a1, __int64 a2, __int64 a3, __int64 **a4)
{
  __int64 *v5; // r12

  if ( a4 )
    v5 = *a4;
  else
    v5 = *(__int64 **)(a2 + 32);
  for ( ; *(__int64 **)(a2 + 40) != v5; v5 += 4 )
  {
    if ( (*(_QWORD *)(v5[1] + 8LL * (a1 >> 6)) & (1LL << a1)) != 0 && !sub_13A0E30(a3, *v5) )
      break;
  }
  return v5;
}
