// Function: sub_BC2DA0
// Address: 0xbc2da0
//
int __fastcall sub_BC2DA0(pthread_rwlock_t *rwlock, __int64 a2)
{
  int v2; // eax
  void (*v3)(); // rax
  void (*v4)(); // r12
  void (*v5)(); // rbx

  while ( &_pthread_key_create )
  {
    v2 = pthread_rwlock_rdlock(rwlock);
    if ( v2 != 11 )
    {
      if ( v2 == 35 )
        sub_4264C5(0x23u);
      break;
    }
  }
  LODWORD(v3) = rwlock[1].__writer;
  if ( (_DWORD)v3 )
  {
    v3 = (void (*)())*(&rwlock[1].__align + 2);
    v4 = (void (*)())((char *)v3 + 16 * *((unsigned int *)&rwlock[1].__align + 8));
    if ( v3 != v4 )
    {
      while ( 1 )
      {
        v5 = v3;
        if ( *(_QWORD *)v3 != -4096 && *(_QWORD *)v3 != -8192 )
          break;
        v3 = (void (*)())((char *)v3 + 16);
        if ( v4 == v3 )
          goto LABEL_5;
      }
      if ( v3 != v4 )
      {
        v3 = *(void (**)())(*(_QWORD *)a2 + 24LL);
        if ( v3 != nullsub_87 )
          goto LABEL_21;
        while ( 1 )
        {
          v5 = (void (*)())((char *)v5 + 16);
          if ( v5 == v4 )
            break;
          while ( 1 )
          {
            v3 = *(void (**)())v5;
            if ( *(_QWORD *)v5 != -8192 && v3 != (void (*)())-4096LL )
              break;
            v5 = (void (*)())((char *)v5 + 16);
            if ( v4 == v5 )
              goto LABEL_5;
          }
          if ( v4 == v5 )
            break;
          v3 = *(void (**)())(*(_QWORD *)a2 + 24LL);
          if ( v3 != nullsub_87 )
LABEL_21:
            LODWORD(v3) = ((__int64 (__fastcall *)(__int64, _QWORD))v3)(a2, *((_QWORD *)v5 + 1));
        }
      }
    }
  }
LABEL_5:
  if ( &_pthread_key_create )
    LODWORD(v3) = pthread_rwlock_unlock(rwlock);
  return (int)v3;
}
