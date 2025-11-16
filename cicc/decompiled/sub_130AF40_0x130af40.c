// Function: sub_130AF40
// Address: 0x130af40
//
__int64 __fastcall sub_130AF40(__int64 a1)
{
  __int64 v2; // rdi
  pthread_mutexattr_t attr; // [rsp+Ch] [rbp-24h] BYREF

  v2 = a1 + 8;
  *(_OWORD *)(v2 - 8) = 0;
  *(_OWORD *)(v2 + 8) = 0;
  *(_OWORD *)(v2 + 24) = 0;
  *(_OWORD *)(v2 + 40) = 0;
  sub_130B140(v2, &unk_42858C8);
  sub_130B140(a1, &unk_42858C8);
  *(_QWORD *)(a1 + 48) = 0;
  if ( pthread_mutexattr_init(&attr) )
    return 1;
  pthread_mutexattr_settype(&attr, 0);
  if ( pthread_mutex_init((pthread_mutex_t *)(a1 + 64), &attr) )
  {
    pthread_mutexattr_destroy(&attr);
    return 1;
  }
  else
  {
    pthread_mutexattr_destroy(&attr);
    return 0;
  }
}
