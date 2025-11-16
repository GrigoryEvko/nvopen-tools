// Function: sub_211C140
// Address: 0x211c140
//
unsigned __int64 __fastcall sub_211C140(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  char *v10; // rax
  int v11; // esi
  char v12; // al
  __int64 v13; // rax
  unsigned __int64 v14; // rdx

  v10 = *(char **)(a2 + 40);
  v11 = 204;
  v12 = *v10;
  if ( v12 != 9 )
  {
    v11 = 205;
    if ( v12 != 10 )
    {
      v11 = 206;
      if ( v12 != 11 )
      {
        v11 = 207;
        if ( v12 != 12 )
        {
          v11 = 462;
          if ( v12 == 13 )
            v11 = 208;
        }
      }
    }
  }
  v13 = sub_200DF30(a1, v11, a2, 0);
  return sub_200D960(a1, v13, v14, a3, a4, a5, a6, a7);
}
