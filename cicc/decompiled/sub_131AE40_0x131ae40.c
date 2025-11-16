// Function: sub_131AE40
// Address: 0x131ae40
//
__int64 sub_131AE40()
{
  __int64 (*v1)(void); // rax

  if ( !byte_4F96B90[0] || qword_4F96B88 )
    return 0;
  v1 = (__int64 (*)(void))dlsym((void *)0xFFFFFFFFFFFFFFFFLL, "pthread_create");
  if ( !v1 )
    v1 = (__int64 (*)(void))&pthread_create;
  qword_4F96B88 = v1;
  return 0;
}
