// Function: sub_16CC820
// Address: 0x16cc820
//
__int64 __fastcall sub_16CC820(
        struct sigaction *a1,
        struct sigaction *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  struct sigaction *v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rcx
  char **v9; // rbx
  __int64 v10; // r8
  __int64 v11; // r9
  char *v12; // rax
  signed __int64 v13; // rax
  pthread_mutex_t **v14; // rdi
  char *s[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v17[6]; // [rsp+10h] [rbp-30h] BYREF

  v6 = a2;
  if ( !qword_4FA0630 )
  {
    a2 = (struct sigaction *)sub_16CC100;
    sub_16C1EA0((__int64)&qword_4FA0630, sub_16CC100, (__int64)sub_16CC450, a4, a5, a6);
  }
  if ( a1 )
  {
    a2 = a1;
    s[0] = (char *)v17;
    sub_16CC110((__int64 *)s, a1, (__int64)v6 + (_QWORD)a1);
  }
  else
  {
    s[1] = 0;
    s[0] = (char *)v17;
    LOBYTE(v17[0]) = 0;
  }
  v9 = (char **)sub_22077B0(16);
  if ( v9 )
  {
    v12 = strdup(s[0]);
    v9[1] = 0;
    *v9 = v12;
  }
  v13 = _InterlockedCompareExchange64(&qword_4FA1088, (signed __int64)v9, 0);
  if ( v13 )
  {
    v7 = v13;
    v8 = 0;
    do
      v7 = _InterlockedCompareExchange64((volatile signed __int64 *)(v7 + 8), (signed __int64)v9, 0);
    while ( v7 );
  }
  v14 = (pthread_mutex_t **)s[0];
  if ( (_QWORD *)s[0] != v17 )
  {
    a2 = (struct sigaction *)(v17[0] + 1LL);
    j_j___libc_free_0(s[0], v17[0] + 1LL);
  }
  sub_16CC1C0(v14, a2, v7, v8, v10, v11);
  return 0;
}
