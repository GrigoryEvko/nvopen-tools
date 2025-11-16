// Function: sub_2B1C020
// Address: 0x2b1c020
//
char *__fastcall sub_2B1C020(
        unsigned int *src,
        unsigned int *a2,
        unsigned int *a3,
        unsigned int *a4,
        _DWORD *a5,
        __int64 a6,
        __int64 *a7)
{
  unsigned int *v9; // r12
  unsigned int v11; // eax
  unsigned int v12; // eax
  signed __int64 v13; // r8
  char *v14; // rbx

  v9 = src;
  if ( src != a2 )
  {
    while ( a3 != a4 )
    {
      if ( sub_2B1BC20(&a7, *a3, *v9) )
      {
        v11 = *a3;
        ++a5;
        ++a3;
        *(a5 - 1) = v11;
        if ( v9 == a2 )
          break;
      }
      else
      {
        v12 = *v9++;
        *a5++ = v12;
        if ( v9 == a2 )
          break;
      }
    }
  }
  v13 = (char *)a2 - (char *)v9;
  if ( a2 != v9 )
  {
    memmove(a5, v9, (char *)a2 - (char *)v9);
    v13 = (char *)a2 - (char *)v9;
  }
  v14 = (char *)a5 + v13;
  if ( a4 != a3 )
    memmove(v14, a3, (char *)a4 - (char *)a3);
  return &v14[(char *)a4 - (char *)a3];
}
