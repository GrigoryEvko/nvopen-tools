// Function: sub_C8C850
// Address: 0xc8c850
//
unsigned int __fastcall sub_C8C850(_BYTE *a1, __int64 a2)
{
  unsigned int result; // eax
  pthread_mutex_t *v3; // r15
  __int64 i; // rbx
  volatile __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v8[8]; // [rsp+10h] [rbp-40h] BYREF

  if ( !a1 )
  {
    v7[1] = 0;
    v7[0] = (__int64)v8;
    LOBYTE(v8[0]) = 0;
    result = (unsigned int)qword_4F84BB0;
    if ( qword_4F84BB0 )
      goto LABEL_3;
LABEL_18:
    result = sub_C7D570((__int64 *)&qword_4F84BB0, (__int64 (*)(void))sub_BC3580, (__int64)sub_BC3540);
    goto LABEL_3;
  }
  v7[0] = (__int64)v8;
  sub_C8B520(v7, a1, (__int64)&a1[a2]);
  result = (unsigned int)qword_4F84BB0;
  if ( !qword_4F84BB0 )
    goto LABEL_18;
LABEL_3:
  v3 = qword_4F84BB0;
  if ( &_pthread_key_create )
  {
    result = pthread_mutex_lock(qword_4F84BB0);
    if ( result )
      sub_4264C5(result);
  }
  for ( i = qword_4F84BA8; i; i = *(_QWORD *)(i + 8) )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)i;
      if ( *(_QWORD *)i )
      {
        result = sub_2241AC0(v7, v5);
        if ( !result )
        {
          v6 = _InterlockedExchange64((volatile __int64 *)i, 0);
          if ( v6 )
            break;
        }
      }
      i = *(_QWORD *)(i + 8);
      if ( !i )
        goto LABEL_12;
    }
    result = _libc_free(v6, v5);
  }
LABEL_12:
  if ( &_pthread_key_create )
    result = pthread_mutex_unlock(v3);
  if ( (_QWORD *)v7[0] != v8 )
    return j_j___libc_free_0(v7[0], v8[0] + 1LL);
  return result;
}
