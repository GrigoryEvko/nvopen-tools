// Function: sub_220E780
// Address: 0x220e780
//
__int64 __fastcall sub_220E780(char *s)
{
  const char *v2; // r12
  __int64 result; // rax
  iconv_t v4; // r13
  size_t v5; // rbp
  iconv_t v6; // rax
  void *v7; // rbp
  size_t v8; // rbx
  char v9; // [rsp+1Eh] [rbp-5Ah] BYREF
  unsigned __int8 v10; // [rsp+1Fh] [rbp-59h] BYREF
  size_t inbytesleft; // [rsp+20h] [rbp-58h] BYREF
  size_t outbytesleft; // [rsp+28h] [rbp-50h] BYREF
  char *inbuf; // [rsp+30h] [rbp-48h] BYREF
  char *outbuf[8]; // [rsp+38h] [rbp-40h] BYREF

  v2 = (const char *)__nl_langinfo_l();
  if ( strcmp(v2, "UTF-8")
    || (result = 32, memcmp(s, &unk_43602EC, 4u))
    && (result = 39, memcmp(s, &unk_43602F0, 4u))
    && (*s != -39 || s[1] != -84 || s[2]) )
  {
    v4 = iconv_open("ASCII//TRANSLIT", v2);
    result = 0;
    if ( v4 != (iconv_t)-1LL )
    {
      inbytesleft = strlen(s);
      inbuf = s;
      outbytesleft = 1;
      outbuf[0] = &v9;
      v5 = iconv(v4, &inbuf, &inbytesleft, outbuf, &outbytesleft);
      iconv_close(v4);
      if ( v5 == -1 )
        return 0;
      v6 = iconv_open(v2, "ASCII");
      v7 = v6;
      if ( v6 == (iconv_t)-1LL )
        return 0;
      inbytesleft = 1;
      inbuf = &v9;
      outbuf[0] = (char *)&v10;
      outbytesleft = 1;
      v8 = iconv(v6, &inbuf, &inbytesleft, outbuf, &outbytesleft);
      iconv_close(v7);
      if ( v8 == -1 )
        return 0;
      else
        return v10;
    }
  }
  return result;
}
