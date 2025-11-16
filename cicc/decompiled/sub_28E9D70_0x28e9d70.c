// Function: sub_28E9D70
// Address: 0x28e9d70
//
void __fastcall sub_28E9D70(__m128i *src, __m128i *a2, char *a3)
{
  char *v5; // r14
  __m128i *v6; // r15
  __m128i *v7; // rdi
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v5 = &a3[(char *)a2 - (char *)src];
  v10 = (char *)a2 - (char *)src;
  v11 = a2 - src;
  if ( (char *)a2 - (char *)src <= 96 )
  {
    sub_28E9CA0(src, a2);
  }
  else
  {
    v6 = src;
    do
    {
      v7 = v6;
      v6 += 7;
      sub_28E9CA0(v7, v6);
    }
    while ( (char *)a2 - (char *)v6 > 96 );
    sub_28E9CA0(v6, a2);
    if ( v10 > 112 )
    {
      v8 = 7;
      do
      {
        sub_28E9830(src->m128i_i8, a2->m128i_i8, a3, v8);
        v9 = 2 * v8;
        v8 *= 4;
        sub_28E9830(a3, v5, src->m128i_i8, v9);
      }
      while ( v11 > v8 );
    }
  }
}
