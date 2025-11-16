// Function: sub_34F1870
// Address: 0x34f1870
//
unsigned int __fastcall sub_34F1870(__int64 a1)
{
  unsigned int result; // eax
  __int64 v2; // [rsp+8h] [rbp-18h] BYREF
  _QWORD v3[2]; // [rsp+10h] [rbp-10h] BYREF

  v3[0] = sub_34F11F0;
  v2 = a1;
  v3[1] = &v2;
  *(_QWORD *)(__readfsqword(0) - 24) = v3;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_97DD80;
  if ( !&_pthread_key_create )
  {
    result = -1;
LABEL_5:
    sub_4264C5(result);
  }
  result = pthread_once(&dword_503BAE8, init_routine);
  if ( result )
    goto LABEL_5;
  return result;
}
