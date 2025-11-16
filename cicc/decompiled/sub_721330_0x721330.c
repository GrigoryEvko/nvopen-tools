// Function: sub_721330
// Address: 0x721330
//
FILE *__fastcall sub_721330(int a1)
{
  const char *v1; // r12
  size_t v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  char v5; // al
  const char *v6; // r14
  const char *v7; // r15
  int v8; // ebx
  unsigned __int64 v9; // r13
  __pid_t v10; // eax
  __int64 v11; // r8
  const char *v12; // rax
  FILE *v13; // rax
  char *v15; // rax
  FILE *v16; // [rsp+8h] [rbp-168h]
  struct stat stat_buf; // [rsp+10h] [rbp-160h] BYREF
  char s[208]; // [rsp+A0h] [rbp-D0h] BYREF

  v1 = qword_4F07918;
  if ( qword_4F07918 || (v15 = getenv("TMPDIR"), qword_4F07918 = v15, (v1 = v15) != 0) && *v15 )
  {
    v2 = strlen(v1);
    v3 = v2 + 24;
    v4 = v2 - 1;
  }
  else
  {
    v1 = "/tmp";
    v4 = 3;
    v3 = 28;
    qword_4F07918 = "/tmp";
  }
  v5 = v1[v4];
  v6 = "/";
  v7 = "w+b";
  if ( v5 == 47 )
    v6 = byte_3F871B3;
  v8 = 21;
  if ( !a1 )
    v7 = "w+";
  v9 = v3 + (v5 != 47);
  while ( 1 )
  {
    if ( v9 > 0x96 )
      sub_685220(0xA4u, (__int64)qword_4F07918);
    v10 = getpid();
    v11 = qword_4F07910++;
    sprintf(s, "%s%sedg%lu_%ld", qword_4F07918, v6, v11, v10);
    if ( __xstat(1, s, &stat_buf) )
    {
      v12 = (const char *)sub_7212A0((__int64)s);
      v13 = fopen(v12, v7);
      if ( v13 )
        break;
    }
    if ( !--v8 )
      sub_685D60(0, 1512, (__int64)s, 9u);
  }
  v16 = v13;
  unlink(s);
  return v16;
}
