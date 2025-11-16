// Function: sub_B60B70
// Address: 0xb60b70
//
char *__fastcall sub_B60B70(int a1)
{
  unsigned int v1; // eax
  char *v2; // r12
  char v4; // [rsp+7h] [rbp-19h] BYREF
  char *v5; // [rsp+8h] [rbp-18h] BYREF

  v5 = &v4;
  *(_QWORD *)(__readfsqword(0) - 24) = &v5;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_B5B9E0;
  if ( !&_pthread_key_create )
  {
    v1 = -1;
LABEL_7:
    sub_4264C5(v1);
  }
  v1 = pthread_once(&dword_4F818F8, init_routine);
  if ( v1 )
    goto LABEL_7;
  v2 = (&off_4B91180)[a1];
  if ( v2 )
    strlen((&off_4B91180)[a1]);
  return v2;
}
