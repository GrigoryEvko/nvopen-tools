// Function: sub_225F3E0
// Address: 0x225f3e0
//
__int64 __fastcall sub_225F3E0(__int64 a1, int a2, unsigned int a3, const char **a4, __m128i a5)
{
  __int64 *v5; // r13
  struct __jmp_buf_tag *v6; // r12
  char *v7; // r12
  unsigned __int64 v8; // rdx
  void *v10; // rdi
  char *s; // [rsp+28h] [rbp-28h] BYREF

  v5 = sub_CEACC0();
  v6 = (struct __jmp_buf_tag *)sub_C94E20((__int64)v5);
  if ( !v6 )
  {
    v10 = (void *)sub_CEECD0(200, 8u);
    memset(v10, 0, 0xC8u);
    sub_C94E10((__int64)v5, v10);
    v6 = (struct __jmp_buf_tag *)sub_C94E20((__int64)v5);
  }
  if ( !_setjmp(v6) )
    return sub_225D540(a1, a2, a3, a4, a5);
  s = 0;
  sub_CEAF80((__int64 *)&s);
  v7 = s;
  v8 = strlen(s);
  if ( v8 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 88) )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)(a1 + 80), v7, v8);
  if ( s )
    j_j___libc_free_0_0((unsigned __int64)s);
  return 9;
}
