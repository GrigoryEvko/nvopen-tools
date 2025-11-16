// Function: sub_307A780
// Address: 0x307a780
//
void __fastcall sub_307A780(unsigned __int64 a1)
{
  unsigned int v1; // eax
  int *v2; // rbx
  int *v3; // r13
  int *v4; // rax
  bool v5; // al
  int *v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rdx
  int *v10; // rdi
  int *v11; // rax
  _QWORD *v12; // r15
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // r14
  _QWORD *v15; // rdi

  if ( !byte_502D1E8 && (unsigned int)sub_2207590((__int64)&byte_502D1E8) )
  {
    stru_502D200.__list.__next = 0;
    *((_OWORD *)&stru_502D200.__align + 1) = 0;
    *(_OWORD *)&stru_502D200.__lock = 0;
    stru_502D200.__kind = 1;
    dword_502D228 = 0;
    dword_502D238 = 0;
    qword_502D240 = 0;
    qword_502D248 = (__int64)&dword_502D238;
    qword_502D250 = (__int64)&dword_502D238;
    qword_502D258 = 0;
    __cxa_atexit((void (*)(void *))sub_307A770, &dword_502D238 - 14, &qword_4A427C0);
    sub_2207640((__int64)&byte_502D1E8);
  }
  if ( &_pthread_key_create )
  {
    v1 = pthread_mutex_lock(&stru_502D200);
    if ( v1 )
      sub_4264C5(v1);
  }
  if ( qword_502D240 )
  {
    v2 = (int *)qword_502D240;
    v3 = &dword_502D238;
    while ( 1 )
    {
      while ( *((_QWORD *)v2 + 4) < a1 )
      {
        v2 = (int *)*((_QWORD *)v2 + 3);
        if ( !v2 )
          goto LABEL_10;
      }
      v4 = (int *)*((_QWORD *)v2 + 2);
      if ( *((_QWORD *)v2 + 4) <= a1 )
        break;
      v3 = v2;
      v2 = (int *)*((_QWORD *)v2 + 2);
      if ( !v4 )
      {
LABEL_10:
        v5 = v3 == &dword_502D238;
        goto LABEL_11;
      }
    }
    v6 = (int *)*((_QWORD *)v2 + 3);
    if ( v6 )
    {
      do
      {
        while ( 1 )
        {
          v7 = *((_QWORD *)v6 + 2);
          v8 = *((_QWORD *)v6 + 3);
          if ( *((_QWORD *)v6 + 4) > a1 )
            break;
          v6 = (int *)*((_QWORD *)v6 + 3);
          if ( !v8 )
            goto LABEL_23;
        }
        v3 = v6;
        v6 = (int *)*((_QWORD *)v6 + 2);
      }
      while ( v7 );
    }
LABEL_23:
    while ( v4 )
    {
      while ( 1 )
      {
        v9 = *((_QWORD *)v4 + 3);
        if ( *((_QWORD *)v4 + 4) >= a1 )
          break;
        v4 = (int *)*((_QWORD *)v4 + 3);
        if ( !v9 )
          goto LABEL_26;
      }
      v2 = v4;
      v4 = (int *)*((_QWORD *)v4 + 2);
    }
LABEL_26:
    if ( (int *)qword_502D248 == v2 && v3 == &dword_502D238 )
      goto LABEL_13;
    if ( v2 != v3 )
    {
      do
      {
        v10 = v2;
        v2 = (int *)sub_220EF30((__int64)v2);
        v11 = sub_220F330(v10, &dword_502D238);
        v12 = (_QWORD *)*((_QWORD *)v11 + 7);
        v13 = (unsigned __int64)v11;
        while ( v12 )
        {
          v14 = (unsigned __int64)v12;
          sub_307A4A0(v12[3]);
          v15 = (_QWORD *)v12[7];
          v12 = (_QWORD *)v12[2];
          sub_307A420(v15);
          j_j___libc_free_0(v14);
        }
        j_j___libc_free_0(v13);
        --qword_502D258;
      }
      while ( v2 != v3 );
      if ( &_pthread_key_create )
        goto LABEL_15;
      return;
    }
  }
  else
  {
    v5 = 1;
    v3 = &dword_502D238;
LABEL_11:
    if ( (int *)qword_502D248 == v3 && v5 )
    {
LABEL_13:
      sub_307A6F0((_QWORD *)qword_502D240);
      qword_502D248 = (__int64)&dword_502D238;
      qword_502D240 = 0;
      qword_502D250 = (__int64)&dword_502D238;
      qword_502D258 = 0;
    }
  }
  if ( &_pthread_key_create )
LABEL_15:
    pthread_mutex_unlock(&stru_502D200);
}
