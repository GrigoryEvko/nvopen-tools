// Function: sub_351BCC0
// Address: 0x351bcc0
//
char *__fastcall sub_351BCC0(_BYTE *src, _BYTE *a2, _BYTE *a3, __int64 a4, __int64 a5, void *a6, __int64 a7)
{
  _BYTE *v9; // r10
  size_t v10; // rbx
  signed __int64 v12; // rbx

  if ( a4 > a5 && a5 <= a7 )
  {
    v9 = src;
    if ( a5 )
    {
      v10 = a3 - a2;
      if ( a2 != a3 )
        memmove(a6, a2, a3 - a2);
      if ( src != a2 )
        memmove(&a3[-(a2 - src)], src, a2 - src);
      if ( v10 )
        memmove(src, a6, v10);
      return &src[v10];
    }
    return v9;
  }
  if ( a4 <= a7 )
  {
    v9 = a3;
    if ( a4 )
    {
      v12 = a2 - src;
      if ( src != a2 )
        memmove(a6, src, a2 - src);
      if ( a2 != a3 )
        memmove(src, a2, a3 - a2);
      v9 = &a3[-v12];
      if ( v12 )
        return (char *)memmove(&a3[-v12], a6, a2 - src);
    }
    return v9;
  }
  return sub_3512D10(src, a2, a3);
}
