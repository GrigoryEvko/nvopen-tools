// Function: sub_8178D0
// Address: 0x8178d0
//
void __fastcall sub_8178D0(__int64 a1, const __m128i *a2, __int64 *a3)
{
  __int64 i; // rdi

  if ( a2 )
  {
    if ( a2[10].m128i_i8[13] == 10 )
    {
      for ( i = a2[11].m128i_i64[0]; i; i = sub_80E180(i, a3) )
      {
        while ( *(char *)(i + 170) < 0 )
        {
          i = *(_QWORD *)(i + 120);
          if ( !i )
            return;
        }
      }
    }
    else
    {
      sub_80D8A0(a2, 1u, 0, a3);
    }
  }
  else
  {
    sub_817850(a1, 1u, a3);
  }
}
