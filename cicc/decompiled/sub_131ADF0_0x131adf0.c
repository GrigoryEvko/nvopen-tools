// Function: sub_131ADF0
// Address: 0x131adf0
//
void sub_131ADF0()
{
  __int64 (*v0)(void); // rax

  if ( !qword_4F96B88 )
  {
    v0 = (__int64 (*)(void))dlsym((void *)0xFFFFFFFFFFFFFFFFLL, "pthread_create");
    if ( !v0 )
      v0 = (__int64 (*)(void))&pthread_create;
    qword_4F96B88 = v0;
  }
}
