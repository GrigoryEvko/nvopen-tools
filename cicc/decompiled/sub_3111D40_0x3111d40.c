// Function: sub_3111D40
// Address: 0x3111d40
//
__int64 sub_3111D40()
{
  unsigned int v0; // eax
  char v2; // [rsp+7h] [rbp-9h] BYREF
  char *v3; // [rsp+8h] [rbp-8h] BYREF

  v3 = &v2;
  *(_QWORD *)(__readfsqword(0) - 24) = &v3;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_3112550;
  if ( !&_pthread_key_create )
  {
    v0 = -1;
LABEL_5:
    sub_4264C5(v0);
  }
  v0 = pthread_once(&dword_5031F40, init_routine);
  if ( v0 )
    goto LABEL_5;
  return unk_5031F48;
}
