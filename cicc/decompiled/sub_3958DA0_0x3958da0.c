// Function: sub_3958DA0
// Address: 0x3958da0
//
char *__fastcall sub_3958DA0(int a1)
{
  unsigned int v1; // eax
  char v3; // [rsp+7h] [rbp-19h] BYREF
  char *v4; // [rsp+8h] [rbp-18h] BYREF

  v4 = &v3;
  *(_QWORD *)(__readfsqword(0) - 24) = &v4;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_3958D80;
  if ( !&_pthread_key_create )
  {
    v1 = -1;
LABEL_5:
    sub_4264C5(v1);
  }
  v1 = pthread_once(&dword_5054868, init_routine);
  if ( v1 )
    goto LABEL_5;
  return &aPrhodwRpfifyMc[qword_4531240[a1]];
}
