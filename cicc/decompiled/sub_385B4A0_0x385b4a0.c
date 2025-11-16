// Function: sub_385B4A0
// Address: 0x385b4a0
//
char *__fastcall sub_385B4A0(char *src, char *a2, unsigned int *a3, unsigned int *a4, _DWORD *a5, _QWORD *a6)
{
  char *v6; // r10
  __int64 v9; // r11
  __int64 v10; // rcx
  signed __int64 v11; // r13
  char *v12; // r8

  v6 = src;
  if ( src != a2 )
  {
    while ( a3 != a4 )
    {
      v9 = *(unsigned int *)v6;
      v10 = *a3;
      if ( *(_QWORD *)(*a6 + 16 * v10) < *(_QWORD *)(*a6 + 16 * v9) )
      {
        *a5 = v10;
        ++a3;
        ++a5;
        if ( v6 == a2 )
          break;
      }
      else
      {
        v6 += 4;
        *a5++ = v9;
        if ( v6 == a2 )
          break;
      }
    }
  }
  v11 = a2 - v6;
  if ( a2 != v6 )
    a5 = memmove(a5, v6, a2 - v6);
  v12 = (char *)a5 + v11;
  if ( a4 != a3 )
    v12 = (char *)memmove(v12, a3, (char *)a4 - (char *)a3);
  return &v12[(char *)a4 - (char *)a3];
}
