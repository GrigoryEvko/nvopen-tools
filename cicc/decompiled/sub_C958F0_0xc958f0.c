// Function: sub_C958F0
// Address: 0xc958f0
//
pthread_t __fastcall sub_C958F0(void *(*start_routine)(void *), void *arg, __int64 a3)
{
  unsigned int v3; // eax
  unsigned int v4; // eax
  pthread_t v5; // r13
  unsigned int v6; // eax
  unsigned int v8; // eax
  pthread_t newthread; // [rsp+18h] [rbp-68h] BYREF
  pthread_attr_t attr; // [rsp+20h] [rbp-60h] BYREF

  v3 = pthread_attr_init(&attr);
  if ( v3 )
    sub_C94E30("pthread_attr_init failed", v3);
  if ( BYTE4(a3) )
  {
    v8 = pthread_attr_setstacksize(&attr, (unsigned int)a3);
    if ( v8 )
      sub_C94E30("pthread_attr_setstacksize failed", v8);
  }
  v4 = pthread_create(&newthread, &attr, start_routine, arg);
  if ( v4 )
    sub_C94E30("pthread_create failed", v4);
  v5 = newthread;
  v6 = pthread_attr_destroy(&attr);
  if ( v6 )
    sub_C94E30("pthread_attr_destroy failed", v6);
  return v5;
}
