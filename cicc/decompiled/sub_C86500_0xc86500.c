// Function: sub_C86500
// Address: 0xc86500
//
int sub_C86500()
{
  const char *v1; // rdi
  int v2; // eax
  int v3; // r12d
  int v4; // ebx
  __int64 v5; // r12
  __pid_t v6; // eax
  __pid_t v7; // [rsp-34h] [rbp-34h] BYREF
  __int64 v8[6]; // [rsp-30h] [rbp-30h] BYREF

  if ( !byte_4F84118 && (unsigned int)sub_2207590(&byte_4F84118) )
  {
    v1 = "/dev/urandom";
    v2 = open("/dev/urandom", 0);
    v3 = v2;
    if ( v2 == -1 || (v4 = read(v2, v8, 4u), close(v3), v1 = (const char *)LODWORD(v8[0]), v4 != 4) )
    {
      v5 = sub_220F850(v1);
      v6 = getpid();
      v8[0] = v5;
      v7 = v6;
      LODWORD(v1) = sub_C86480(v8, &v7);
    }
    srand((unsigned int)v1);
    sub_2207640(&byte_4F84118);
  }
  return rand();
}
