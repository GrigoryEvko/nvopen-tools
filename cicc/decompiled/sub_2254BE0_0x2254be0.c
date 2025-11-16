// Function: sub_2254BE0
// Address: 0x2254be0
//
__int64 __fastcall sub_2254BE0(
        __int64 a1,
        mbstate_t *a2,
        const char *a3,
        unsigned __int64 a4,
        const char **a5,
        wchar_t *a6,
        unsigned __int64 a7,
        wchar_t **a8)
{
  const char *v12; // rax
  size_t v13; // rdx
  const char *v14; // r11
  size_t v15; // rax
  wchar_t *v16; // rax
  unsigned int v17; // r12d
  mbstate_t v19; // rdx
  wchar_t *i; // rdi
  size_t v21; // rax
  mbstate_t v22; // rax
  const char *v23; // [rsp+0h] [rbp-58h]
  mbstate_t p; // [rsp+18h] [rbp-40h] BYREF

  p = *a2;
  __uselocale();
  *a5 = a3;
  *a8 = a6;
  if ( (unsigned __int64)a3 < a4 )
  {
    while ( (unsigned __int64)a6 < a7 )
    {
      v12 = (const char *)memchr(a3, 0, a4 - (_QWORD)a3);
      v13 = a4 - (_QWORD)a3;
      v14 = v12;
      if ( v12 )
        v13 = v12 - a3;
      else
        v14 = (const char *)a4;
      v23 = v14;
      v15 = mbsnrtowcs(a6, a5, v13, (__int64)(a7 - (_QWORD)a6) >> 2, a2);
      if ( v15 == -1 )
      {
        for ( i = *a8; ; *a8 = i )
        {
          v21 = mbrtowc(i, a3, a4 - (_QWORD)a3, &p);
          if ( v21 > 0xFFFFFFFFFFFFFFFDLL )
            break;
          a3 += v21;
          i = *a8 + 1;
        }
        v22 = p;
        *a5 = a3;
        v17 = 2;
        *a2 = v22;
        goto LABEL_9;
      }
      v16 = &(*a8)[v15];
      if ( *a5 && *a5 < v23 )
      {
        *a8 = v16;
        v17 = 1;
        goto LABEL_9;
      }
      *a5 = v23;
      *a8 = v16;
      if ( (unsigned __int64)v23 >= a4 )
        break;
      if ( (unsigned __int64)v16 >= a7 )
      {
        v17 = 1;
        goto LABEL_9;
      }
      v19 = *a2;
      a3 = v23 + 1;
      a6 = v16 + 1;
      *a5 = v23 + 1;
      p = v19;
      *a8 = v16 + 1;
      *v16 = 0;
      if ( (unsigned __int64)(v23 + 1) >= a4 )
        break;
    }
  }
  v17 = 0;
LABEL_9:
  __uselocale();
  return v17;
}
