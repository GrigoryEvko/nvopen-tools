// Function: sub_22549F0
// Address: 0x22549f0
//
__int64 __fastcall sub_22549F0(
        __int64 a1,
        mbstate_t *a2,
        const wchar_t *a3,
        wchar_t *a4,
        const wchar_t **a5,
        char *a6,
        unsigned __int64 a7,
        void **a8)
{
  wchar_t *v13; // rax
  size_t v14; // rdx
  wchar_t *v15; // r12
  size_t v16; // rax
  char *v17; // rax
  unsigned int v18; // r12d
  size_t v20; // rax
  size_t v21; // r15
  char *v22; // rdi
  wchar_t v23; // esi
  size_t v24; // rax
  mbstate_t ps; // [rsp+28h] [rbp-50h] BYREF
  char s[72]; // [rsp+30h] [rbp-48h] BYREF

  ps = *a2;
  __uselocale();
  *a5 = a3;
  *a8 = a6;
  if ( a3 < a4 )
  {
    while ( (unsigned __int64)a6 < a7 )
    {
      v13 = wmemchr(a3, 0, a4 - a3);
      v14 = a4 - a3;
      v15 = v13;
      if ( v13 )
        v14 = v13 - a3;
      else
        v15 = a4;
      v16 = wcsnrtombs(a6, a5, v14, a7 - (_QWORD)a6, a2);
      if ( v16 == -1 )
      {
        if ( *a5 > a3 )
        {
          v22 = (char *)*a8;
          do
          {
            v23 = *a3++;
            v24 = (size_t)*a8 + wcrtomb(v22, v23, &ps);
            *a8 = (void *)v24;
            v22 = (char *)v24;
          }
          while ( *a5 > a3 );
        }
        v18 = 2;
        *a2 = ps;
        goto LABEL_9;
      }
      v17 = (char *)*a8 + v16;
      if ( *a5 && *a5 < v15 )
      {
        *a8 = v17;
        v18 = 1;
        goto LABEL_9;
      }
      *a5 = v15;
      *a8 = v17;
      if ( v15 >= a4 )
        break;
      ps = *a2;
      v20 = wcrtomb(s, *v15, &ps);
      v21 = v20;
      if ( a7 - (unsigned __int64)*a8 < v20 )
      {
        v18 = 1;
        goto LABEL_9;
      }
      memcpy(*a8, s, v20);
      *a2 = ps;
      a6 = (char *)*a8 + v21;
      a3 = *a5 + 1;
      *a8 = a6;
      *a5 = a3;
      if ( a3 >= a4 )
        break;
    }
  }
  v18 = 0;
LABEL_9:
  __uselocale();
  return v18;
}
