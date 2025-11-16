// Function: sub_ED1630
// Address: 0xed1630
//
__int64 *__fastcall sub_ED1630(__int64 *a1, __int64 a2, unsigned __int64 a3, int a4)
{
  __int64 v6; // rcx
  __int64 v7; // rcx
  __int64 v8; // rdx
  size_t v9; // rax
  __int64 v10; // rbx
  char s[41]; // [rsp+7h] [rbp-29h] BYREF

  *a1 = (__int64)(a1 + 2);
  sub_ED0450(a1, "__profn_", (__int64)"");
  if ( a3 > 0x3FFFFFFFFFFFFFFFLL - a1[1] )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(a1, a2, a3, v6);
  if ( (unsigned int)(a4 - 7) <= 1 )
  {
    v7 = 8;
    v8 = 0;
    strcpy(s, "-:;<>/\"'");
    while ( 1 )
    {
      v10 = sub_22418E0(a1, s, v8, v7);
      if ( v10 == -1 )
        break;
      *(_BYTE *)(*a1 + v10) = 95;
      v9 = strlen(s);
      v8 = v10 + 1;
      v7 = v9;
    }
  }
  return a1;
}
