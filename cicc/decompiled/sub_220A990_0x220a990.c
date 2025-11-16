// Function: sub_220A990
// Address: 0x220a990
//
int __fastcall sub_220A990(volatile signed __int32 **a1)
{
  volatile signed __int32 *v1; // rax

  *a1 = 0;
  sub_220A920();
  v1 = qword_4FD4F50;
  *a1 = qword_4FD4F50;
  if ( v1 != (volatile signed __int32 *)unk_4FD4F58 )
  {
    if ( !byte_4FD6588 && (unsigned int)sub_2207590((__int64)&byte_4FD6588) )
    {
      stru_4FD65A0.__list.__next = 0;
      *(_OWORD *)&stru_4FD65A0.__lock = 0;
      *((_OWORD *)&stru_4FD65A0.__align + 1) = 0;
      sub_2207640((__int64)&byte_4FD6588);
    }
    if ( &_pthread_key_create && pthread_mutex_lock(&stru_4FD65A0) )
      JUMPOUT(0x42549E);
    v1 = qword_4FD4F50;
    if ( &_pthread_key_create )
    {
      _InterlockedAdd(qword_4FD4F50, 1u);
      v1 = qword_4FD4F50;
    }
    else
    {
      ++*qword_4FD4F50;
    }
    *a1 = v1;
    if ( &_pthread_key_create )
    {
      LODWORD(v1) = pthread_mutex_unlock(&stru_4FD65A0);
      if ( (_DWORD)v1 )
        JUMPOUT(0x425470);
    }
  }
  return (int)v1;
}
