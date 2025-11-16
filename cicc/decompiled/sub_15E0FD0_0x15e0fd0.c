// Function: sub_15E0FD0
// Address: 0x15e0fd0
//
char *__fastcall sub_15E0FD0(int a1)
{
  unsigned int v1; // eax
  char *v2; // r12
  char v4; // [rsp+7h] [rbp-19h] BYREF
  char *v5; // [rsp+8h] [rbp-18h] BYREF

  v5 = &v4;
  *(_QWORD *)(__readfsqword(0) - 24) = &v5;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_15DE590;
  if ( !&_pthread_key_create )
  {
    v1 = -1;
LABEL_7:
    sub_4264C5(v1);
  }
  v1 = pthread_once(&dword_4F9E14C, init_routine);
  if ( v1 )
    goto LABEL_7;
  v2 = (&off_4C6F380)[a1];
  if ( v2 )
    strlen((&off_4C6F380)[a1]);
  return v2;
}
