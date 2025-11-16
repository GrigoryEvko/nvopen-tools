// Function: sub_35EE460
// Address: 0x35ee460
//
char *__fastcall sub_35EE460(int a1)
{
  unsigned int v1; // eax
  char v3; // [rsp+7h] [rbp-19h] BYREF
  char *v4; // [rsp+8h] [rbp-18h] BYREF

  v4 = &v3;
  *(_QWORD *)(__readfsqword(0) - 24) = &v4;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_35ED220;
  if ( !&_pthread_key_create )
  {
    v1 = -1;
LABEL_5:
    sub_4264C5(v1);
  }
  v1 = pthread_once(&dword_5040808, init_routine);
  if ( v1 )
    goto LABEL_5;
  return (char *)&unk_4CE00A0 + word_44F0A80[a1 - 1];
}
