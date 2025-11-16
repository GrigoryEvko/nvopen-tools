// Function: sub_16D5920
// Address: 0x16d5920
//
void __fastcall sub_16D5920(__int64 a1)
{
  __int64 v2; // rbx
  unsigned int v3; // eax
  pthread_mutex_t *v4; // r13
  __int64 *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rax
  _QWORD *v10; // r13
  __int64 *v11; // rdi
  __int64 v12; // r13
  __int64 (__fastcall *v13)(__int64); // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  pthread_mutex_t *v18; // r13
  char v19; // [rsp+Fh] [rbp-B1h] BYREF
  __int64 v20; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v21; // [rsp+18h] [rbp-A8h] BYREF
  pthread_mutex_t **p_mutex; // [rsp+20h] [rbp-A0h] BYREF
  char *v23; // [rsp+28h] [rbp-98h] BYREF
  __int64 v24; // [rsp+30h] [rbp-90h] BYREF
  __int64 *v25; // [rsp+38h] [rbp-88h]
  _QWORD v26[2]; // [rsp+40h] [rbp-80h] BYREF
  pthread_mutex_t *mutex; // [rsp+50h] [rbp-70h] BYREF
  __int64 *v28; // [rsp+58h] [rbp-68h]
  __int64 (__fastcall *v29)(const __m128i **, const __m128i *, int); // [rsp+60h] [rbp-60h]
  _QWORD *(__fastcall *v30)(_QWORD *, __int64 **, __int64); // [rsp+68h] [rbp-58h]
  _QWORD *v31; // [rsp+70h] [rbp-50h] BYREF
  __int64 *v32; // [rsp+78h] [rbp-48h]
  pthread_mutex_t ***p_p_mutex; // [rsp+80h] [rbp-40h]
  char **v34; // [rsp+88h] [rbp-38h]

  while ( 1 )
  {
    v2 = *(_QWORD *)(a1 + 8);
    LOBYTE(v28) = 0;
    v24 = 0;
    v25 = 0;
    mutex = (pthread_mutex_t *)(v2 + 104);
    if ( &_pthread_key_create )
    {
      v3 = pthread_mutex_lock((pthread_mutex_t *)(v2 + 104));
      if ( v3 )
        goto LABEL_45;
      v2 = *(_QWORD *)(a1 + 8);
    }
    LOBYTE(v28) = 1;
    if ( !*(_BYTE *)(v2 + 284) )
      goto LABEL_21;
    do
    {
      if ( *(_QWORD *)(v2 + 72) != *(_QWORD *)(v2 + 40) )
        break;
      sub_2210B30(v2 + 144, &mutex);
    }
    while ( *(_BYTE *)(v2 + 284) );
    v2 = *(_QWORD *)(a1 + 8);
    if ( !*(_BYTE *)(v2 + 284) )
    {
LABEL_21:
      if ( *(_QWORD *)(v2 + 72) == *(_QWORD *)(v2 + 40) )
        break;
    }
    v4 = (pthread_mutex_t *)(v2 + 192);
    if ( &_pthread_key_create )
    {
      v3 = pthread_mutex_lock((pthread_mutex_t *)(v2 + 192));
      if ( v3 )
        goto LABEL_45;
      v2 = *(_QWORD *)(a1 + 8);
    }
    _InterlockedAdd((volatile signed __int32 *)(v2 + 280), 1u);
    if ( &_pthread_key_create )
      pthread_mutex_unlock(v4);
    v5 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 40LL);
    v6 = *v5;
    v7 = v5[1];
    *v5 = 0;
    v5[1] = 0;
    v8 = v24;
    v24 = v6;
    v31 = (_QWORD *)v8;
    v9 = v25;
    v25 = (__int64 *)v7;
    v32 = v9;
    sub_16D4F80((__int64 *)&v31);
    v10 = *(_QWORD **)(a1 + 8);
    v11 = (__int64 *)v10[5];
    if ( v11 == (__int64 *)(v10[7] - 16LL) )
    {
      sub_16D4F80(v11);
      j_j___libc_free_0(v10[6], 512);
      v14 = (__int64 *)(v10[8] + 8LL);
      v10[8] = v14;
      v15 = *v14;
      v16 = *v14 + 512;
      v10[6] = v15;
      v10[7] = v16;
      v10[5] = v15;
    }
    else
    {
      sub_16D4F80(v11);
      v10[5] += 16LL;
    }
    if ( (_BYTE)v28 && mutex && &_pthread_key_create )
      pthread_mutex_unlock(mutex);
    v12 = v24;
    if ( !v24 )
      sub_42641C(3u);
    v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v24 + 32LL);
    if ( v13 == sub_16D44A0 )
    {
      v20 = v24;
      mutex = (pthread_mutex_t *)(v24 + 32);
      v28 = &v20;
      v19 = 0;
      v30 = sub_16D4420;
      p_mutex = &mutex;
      v29 = sub_16D4240;
      v23 = &v19;
      v21 = v24;
      v26[0] = sub_16D4680;
      v31 = v26;
      v32 = &v21;
      p_p_mutex = &p_mutex;
      v26[1] = 0;
      v34 = &v23;
      *(_QWORD *)(__readfsqword(0) - 24) = &v31;
      *(_QWORD *)(__readfsqword(0) - 32) = sub_16D42A0;
      if ( !&_pthread_key_create )
      {
        v3 = -1;
LABEL_45:
        sub_4264C5(v3);
      }
      v3 = pthread_once((pthread_once_t *)(v12 + 24), init_routine);
      if ( v3 )
        goto LABEL_45;
      if ( !v19 )
        sub_42641C(2u);
      if ( _InterlockedExchange((volatile __int32 *)(v12 + 16), 1) < 0 )
        sub_222D1B0();
      if ( v29 )
        v29((const __m128i **)&mutex, (const __m128i *)&mutex, 3);
    }
    else
    {
      v13(v24);
    }
    v17 = *(_QWORD *)(a1 + 8);
    v18 = (pthread_mutex_t *)(v17 + 192);
    if ( &_pthread_key_create )
    {
      v3 = pthread_mutex_lock((pthread_mutex_t *)(v17 + 192));
      if ( v3 )
        goto LABEL_45;
      v17 = *(_QWORD *)(a1 + 8);
    }
    _InterlockedSub((volatile signed __int32 *)(v17 + 280), 1u);
    if ( &_pthread_key_create )
      pthread_mutex_unlock(v18);
    sub_2210B70(*(_QWORD *)(a1 + 8) + 232LL);
    sub_16D4F80(&v24);
  }
  if ( (_BYTE)v28 && mutex && &_pthread_key_create )
    pthread_mutex_unlock(mutex);
  sub_16D4F80(&v24);
}
