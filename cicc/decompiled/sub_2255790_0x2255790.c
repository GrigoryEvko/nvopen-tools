// Function: sub_2255790
// Address: 0x2255790
//
__int64 __fastcall sub_2255790(__int64 a1, const char **a2, volatile signed __int32 **a3)
{
  const char *v4; // rax
  __int128 *v5; // r13
  volatile signed __int32 *v7[6]; // [rsp+8h] [rbp-30h] BYREF

  sub_222FD10(a3, (__int64)a2);
  v4 = (const char *)__nl_langinfo_l();
  bind_textdomain_codeset(*a2, v4);
  v5 = sub_2254490();
  sub_2208E20(v7, a3);
  LODWORD(v5) = sub_22546A0((__int64)v5, *a2, v7);
  sub_2209150(v7);
  return (unsigned int)v5;
}
