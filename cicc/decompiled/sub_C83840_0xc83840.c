// Function: sub_C83840
// Address: 0xc83840
//
__int64 __fastcall sub_C83840(_QWORD *a1)
{
  char *v1; // r14
  char *pw_dir; // r13
  size_t v4; // rax
  __int64 v5; // rdi
  size_t v6; // r12
  unsigned int v7; // r12d
  signed __int64 v9; // r12
  char *v10; // rax
  __uid_t v11; // eax
  struct passwd *result; // [rsp+8h] [rbp-58h] BYREF
  struct passwd resultbuf; // [rsp+10h] [rbp-50h] BYREF

  v1 = 0;
  pw_dir = getenv("HOME");
  if ( pw_dir )
    goto LABEL_2;
  v9 = sysconf(70);
  if ( v9 <= 0 )
    v9 = 0x4000;
  v10 = (char *)sub_2207820(v9);
  v1 = v10;
  if ( v10 )
    memset(v10, 0, v9);
  result = 0;
  v11 = getuid();
  getpwuid_r(v11, &resultbuf, v1, v9, &result);
  if ( result )
  {
    pw_dir = result->pw_dir;
    if ( pw_dir )
    {
LABEL_2:
      a1[1] = 0;
      v4 = strlen(pw_dir);
      v5 = 0;
      v6 = v4;
      if ( v4 > a1[2] )
      {
        sub_C8D290(a1, a1 + 3, v4, 1);
        v5 = a1[1];
        if ( !v6 )
          goto LABEL_5;
      }
      else if ( !v4 )
      {
LABEL_5:
        a1[1] = v5 + v6;
        v7 = 1;
        goto LABEL_6;
      }
      memcpy((void *)(*a1 + v5), pw_dir, v6);
      v5 = a1[1];
      goto LABEL_5;
    }
  }
  v7 = 0;
LABEL_6:
  if ( v1 )
    j_j___libc_free_0_0(v1);
  return v7;
}
