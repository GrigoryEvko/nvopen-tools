// Function: sub_1F209D0
// Address: 0x1f209d0
//
char *__fastcall sub_1F209D0(char *src, char *a2, char *a3, char *a4, _DWORD *a5, __int64 a6)
{
  char *v6; // r10
  int v9; // eax
  int v10; // edx
  size_t v11; // r13
  char *v12; // r8

  v6 = src;
  if ( src != a2 )
  {
    while ( a3 != a4 )
    {
      v9 = *(_DWORD *)a3;
      v10 = *(_DWORD *)v6;
      if ( *(_DWORD *)a3 != -1
        && (v10 == -1
         || *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a6 + 232) + 8LL)
                      + 40LL * (unsigned int)(v9 + *(_DWORD *)(*(_QWORD *)(a6 + 232) + 32LL))
                      + 8) > *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a6 + 232) + 8LL)
                                       + 40LL * (unsigned int)(v10 + *(_DWORD *)(*(_QWORD *)(a6 + 232) + 32LL))
                                       + 8)) )
      {
        *a5 = v9;
        a3 += 4;
        ++a5;
        if ( v6 == a2 )
          break;
      }
      else
      {
        v6 += 4;
        *a5++ = v10;
        if ( v6 == a2 )
          break;
      }
    }
  }
  v11 = a2 - v6;
  if ( a2 != v6 )
    a5 = memmove(a5, v6, v11);
  v12 = (char *)a5 + v11;
  if ( a4 != a3 )
    v12 = (char *)memmove(v12, a3, a4 - a3);
  return &v12[a4 - a3];
}
