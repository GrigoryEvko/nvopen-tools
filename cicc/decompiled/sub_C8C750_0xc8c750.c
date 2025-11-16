// Function: sub_C8C750
// Address: 0xc8c750
//
__int64 __fastcall sub_C8C750(_BYTE *a1, __int64 a2)
{
  char **v2; // rbx
  char *v3; // rax
  signed __int64 v4; // rax
  signed __int64 v6; // rdx
  char *s[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v8[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( !qword_4F84150 )
    sub_C7D570(&qword_4F84150, sub_C8B3C0, (__int64)sub_C8C250);
  if ( a1 )
  {
    s[0] = (char *)v8;
    sub_C8B520((__int64 *)s, a1, (__int64)&a1[a2]);
  }
  else
  {
    s[1] = 0;
    s[0] = (char *)v8;
    LOBYTE(v8[0]) = 0;
  }
  v2 = (char **)sub_22077B0(16);
  if ( v2 )
  {
    v3 = strdup(s[0]);
    v2[1] = 0;
    *v2 = v3;
  }
  v4 = _InterlockedCompareExchange64(&qword_4F84BA8, (signed __int64)v2, 0);
  if ( v4 )
  {
    v6 = v4;
    do
      v6 = _InterlockedCompareExchange64((volatile signed __int64 *)(v6 + 8), (signed __int64)v2, 0);
    while ( v6 );
  }
  if ( (_QWORD *)s[0] != v8 )
    j_j___libc_free_0(s[0], v8[0] + 1LL);
  sub_C8BD80();
  return 0;
}
