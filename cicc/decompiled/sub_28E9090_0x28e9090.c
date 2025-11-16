// Function: sub_28E9090
// Address: 0x28e9090
//
char *__fastcall sub_28E9090(_DWORD *src, _BYTE *a2, char *a3, char *a4, _QWORD *a5)
{
  char *v5; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  size_t v9; // r13
  char *v10; // r8

  v5 = a3;
  if ( a4 != a3 && a2 != (_BYTE *)src )
  {
    do
    {
      if ( *((_DWORD *)v5 + 2) > src[2] )
      {
        v7 = *(_QWORD *)v5;
        a5 += 2;
        v5 += 16;
        *(a5 - 2) = v7;
        *((_DWORD *)a5 - 2) = *((_DWORD *)v5 - 2);
        if ( a2 == (_BYTE *)src )
          break;
      }
      else
      {
        v8 = *(_QWORD *)src;
        src += 4;
        a5 += 2;
        *(a5 - 2) = v8;
        *((_DWORD *)a5 - 2) = *(src - 2);
        if ( a2 == (_BYTE *)src )
          break;
      }
    }
    while ( a4 != v5 );
  }
  v9 = a2 - (_BYTE *)src;
  if ( a2 != (_BYTE *)src )
    a5 = memmove(a5, src, v9);
  v10 = (char *)a5 + v9;
  if ( a4 != v5 )
    v10 = (char *)memmove(v10, v5, a4 - v5);
  return &v10[a4 - v5];
}
