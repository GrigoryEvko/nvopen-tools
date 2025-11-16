// Function: sub_12C5770
// Address: 0x12c5770
//
__int64 __fastcall sub_12C5770(__int64 a1, int a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // r13
  struct __jmp_buf_tag *v5; // r12
  char *v6; // r12
  size_t v7; // rdx
  __int64 v8; // rcx
  void *v10; // rdi
  char *s; // [rsp+28h] [rbp-28h] BYREF

  v4 = sub_1C3E710();
  v5 = (struct __jmp_buf_tag *)sub_16D40F0(v4);
  if ( !v5 )
  {
    v10 = (void *)sub_1C42D70(200, 8);
    memset(v10, 0, 0xC8u);
    sub_16D40E0(v4, v10);
    v5 = (struct __jmp_buf_tag *)sub_16D40F0(v4);
  }
  if ( !_setjmp(v5) )
    return sub_12C35D0(a1, a2, a3, a4);
  s = 0;
  sub_1C3E9C0(&s);
  v6 = s;
  v7 = strlen(s);
  if ( v7 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 88) )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(a1 + 80, v6, v7, v8);
  if ( s )
    j_j___libc_free_0_0(s);
  return 9;
}
