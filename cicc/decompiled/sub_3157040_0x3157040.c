// Function: sub_3157040
// Address: 0x3157040
//
void *__fastcall sub_3157040(__int64 a1, unsigned __int64 *a2)
{
  __int64 *v2; // r13
  void *v3; // r12
  unsigned int v4; // eax
  _QWORD *v5; // rsi
  _QWORD *v6; // rdi
  _BYTE *v8; // rsi
  __int64 v9; // rax
  void *v10; // [rsp+0h] [rbp-40h] BYREF
  void *v11; // [rsp+8h] [rbp-38h] BYREF

  v2 = sub_3156D40();
  v3 = sub_3156FF0(a1, a2);
  if ( v3 == &unk_5034368 )
    return v3;
  if ( &_pthread_key_create )
  {
    v4 = pthread_mutex_lock((pthread_mutex_t *)(v2 + 11));
    if ( v4 )
      sub_4264C5(v4);
  }
  v10 = v3;
  if ( a1 )
  {
    v5 = (_QWORD *)v2[4];
    v6 = (_QWORD *)v2[3];
    v11 = v3;
    if ( v5 == sub_3156E70(v6, (__int64)v5, (__int64 *)&v11) )
    {
      v8 = (_BYTE *)v2[4];
      if ( v8 == (_BYTE *)v2[5] )
      {
        sub_16F1140((__int64)(v2 + 3), v8, &v10);
      }
      else
      {
        if ( v8 )
        {
          *(_QWORD *)v8 = v10;
          v8 = (_BYTE *)v2[4];
        }
        v2[4] = (__int64)(v8 + 8);
      }
    }
    else
    {
      nullsub_1757();
    }
    goto LABEL_7;
  }
  if ( v2[6] )
  {
    nullsub_1757();
    v9 = (__int64)v10;
    if ( (void *)v2[6] == v10 )
      goto LABEL_7;
  }
  else
  {
    v9 = (__int64)v3;
  }
  v2[6] = v9;
LABEL_7:
  if ( &_pthread_key_create )
    pthread_mutex_unlock((pthread_mutex_t *)(v2 + 11));
  return v3;
}
