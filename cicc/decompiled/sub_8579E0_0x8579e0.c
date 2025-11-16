// Function: sub_8579E0
// Address: 0x8579e0
//
__int64 *__fastcall sub_8579E0(_QWORD *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r15
  size_t v7; // r12
  const char *v8; // r14
  char v9; // bl
  char *s2; // [rsp+8h] [rbp-38h]

  v6 = 0;
  sub_7B8B50((unsigned __int64)a1, a2, a3, a4, a5, a6);
  *a1 = *(_QWORD *)&dword_4F063F8;
  if ( word_4F06418[0] == 1 )
  {
    if ( dword_4D04788 && qword_4F06400 == 11 )
    {
      if ( !memcmp(qword_4F06410, "__VA_ARGS__", 0xBu) )
        sub_6851C0(0x3C9u, dword_4F07508);
    }
    else if ( unk_4D041B8 && qword_4F06400 == 10 && !memcmp(qword_4F06410, "__VA_OPT__", 0xAu) )
    {
      sub_6851C0(0xB7Bu, dword_4F07508);
    }
    v6 = (__int64 *)unk_4D03E90;
    if ( !unk_4D03E90 )
      return (__int64 *)qword_4D03D40[39];
    v7 = qword_4F06400;
    s2 = (char *)qword_4F06410;
    while ( 1 )
    {
      v8 = (const char *)*(&off_4B6DB80 + *((unsigned __int8 *)v6 + 8));
      v9 = *((_BYTE *)v6 + 8);
      if ( strlen(v8) == v7 && !strncmp(v8, s2, v7) )
        break;
      v6 = (__int64 *)*v6;
      if ( !v6 )
        return (__int64 *)qword_4D03D40[39];
    }
    if ( v9 == 28 )
    {
      sub_7BC390();
      qword_4F06410 = s2;
      if ( !memcmp(qword_4F06460, "diagnostic", 0xAu) )
      {
        v6 = (__int64 *)*v6;
        if ( !v6 )
          return (__int64 *)qword_4D03D40[39];
      }
    }
  }
  return v6;
}
