// Function: sub_1462B80
// Address: 0x1462b80
//
char *__fastcall sub_1462B80(
        __int64 *src,
        __int64 *a2,
        __int64 **a3,
        __int64 **a4,
        _QWORD *a5,
        __int64 a6,
        _QWORD *a7,
        _QWORD *a8,
        __int64 *a9,
        __int64 a10)
{
  __int64 **v10; // r13
  __int64 *v11; // r12
  __int64 **v13; // r14
  int i; // eax
  __int64 *v15; // rax
  __int64 v16; // rax
  char *v17; // r11

  v10 = a3;
  v11 = src;
  if ( src != a2 && a3 != a4 )
  {
    v13 = a3;
    for ( i = sub_1462150(a7, a8, *a9, *a3, *src, a10, 0); ; i = sub_1462150(a7, a8, *a9, *v13, *v11, a10, 0) )
    {
      if ( i < 0 )
      {
        v15 = *v13;
        ++a5;
        ++v13;
        *(a5 - 1) = v15;
        if ( v11 == a2 )
          goto LABEL_9;
      }
      else
      {
        v16 = *v11;
        ++a5;
        ++v11;
        *(a5 - 1) = v16;
        if ( v11 == a2 )
        {
LABEL_9:
          v10 = v13;
          break;
        }
      }
      if ( v13 == a4 )
        goto LABEL_9;
    }
  }
  if ( a2 != v11 )
    memmove(a5, v11, (char *)a2 - (char *)v11);
  v17 = (char *)a5 + (char *)a2 - (char *)v11;
  if ( a4 != v10 )
    v17 = (char *)memmove(v17, v10, (char *)a4 - (char *)v10);
  return &v17[(char *)a4 - (char *)v10];
}
