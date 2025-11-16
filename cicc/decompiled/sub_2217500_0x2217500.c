// Function: sub_2217500
// Address: 0x2217500
//
unsigned __int64 __fastcall sub_2217500(__int64 a1, wint_t *a2, unsigned __int64 a3, char a4, __int64 a5)
{
  wint_t *v8; // rbp
  unsigned __int64 v10; // r15
  __int64 v11; // rdi
  int v12; // eax
  unsigned __int64 v14; // r13
  int v15; // eax

  v8 = a2;
  __uselocale();
  if ( *(_BYTE *)(a1 + 24) )
  {
    if ( (unsigned __int64)a2 < a3 )
    {
      v10 = a5 + ((a3 - 1 - (unsigned __int64)a2) >> 2) + 1;
      do
      {
        while ( 1 )
        {
          v11 = (int)*v8;
          if ( (unsigned int)v11 > 0x7F )
            break;
          ++a5;
          ++v8;
          *(_BYTE *)(a5 - 1) = *(_BYTE *)(a1 + v11 + 25);
          if ( v10 == a5 )
            goto LABEL_9;
        }
        v12 = wctob(v11);
        if ( v12 == -1 )
          LOBYTE(v12) = a4;
        ++a5;
        ++v8;
        *(_BYTE *)(a5 - 1) = v12;
      }
      while ( v10 != a5 );
    }
  }
  else if ( (unsigned __int64)a2 < a3 )
  {
    v14 = a5 + ((a3 - 1 - (unsigned __int64)a2) >> 2) + 1;
    do
    {
      v15 = wctob(*v8);
      if ( v15 == -1 )
        LOBYTE(v15) = a4;
      ++a5;
      ++v8;
      *(_BYTE *)(a5 - 1) = v15;
    }
    while ( a5 != v14 );
  }
LABEL_9:
  __uselocale();
  return a3;
}
