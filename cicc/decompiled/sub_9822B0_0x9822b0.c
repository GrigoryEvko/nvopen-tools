// Function: sub_9822B0
// Address: 0x9822b0
//
void __fastcall sub_9822B0(char *a1, char *a2, __int64 (__fastcall *a3)(__m128i *, const __m128i *))
{
  const __m128i *v4; // rbx
  const __m128i *v5; // rdi

  if ( a2 - a1 <= 1024 )
  {
    sub_982180(a1, a2, a3);
  }
  else
  {
    v4 = (const __m128i *)(a1 + 1024);
    sub_982180(a1, a1 + 1024, a3);
    if ( a2 != a1 + 1024 )
    {
      do
      {
        v5 = v4;
        v4 += 4;
        sub_9820C0(v5, a3);
      }
      while ( a2 != (char *)v4 );
    }
  }
}
