// Function: sub_CA08F0
// Address: 0xca08f0
//
__int64 __fastcall sub_CA08F0(
        __int64 *a1,
        const void *a2,
        size_t a3,
        __int64 a4,
        __int64 a5,
        char a6,
        _BYTE *src,
        size_t n,
        _BYTE *a9,
        __int64 a10)
{
  __int64 result; // rax
  __int64 v12; // rbx
  unsigned int v13; // eax
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r9
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // rax
  unsigned int v22; // r8d
  _QWORD *v23; // r10
  _QWORD *v24; // rcx
  __int64 *v25; // rax
  __int64 *v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // rbx
  _QWORD *v29; // [rsp+0h] [rbp-90h]
  _QWORD *v30; // [rsp+8h] [rbp-88h]
  unsigned int v31; // [rsp+14h] [rbp-7Ch]
  pthread_mutex_t *mutex; // [rsp+18h] [rbp-78h]
  __int64 v33; // [rsp+20h] [rbp-70h]
  __int64 v34; // [rsp+30h] [rbp-60h]
  __int64 v37; // [rsp+50h] [rbp-40h] BYREF
  __int64 *v38; // [rsp+58h] [rbp-38h] BYREF

  result = a10;
  if ( !a6 )
  {
    *a1 = 0;
    return result;
  }
  if ( !qword_4F84F60 )
    sub_C7D570(&qword_4F84F60, (__int64 (*)(void))sub_CA0780, (__int64)sub_C9FD10);
  v12 = qword_4F84F60;
  v38 = &v37;
  v37 = qword_4F84F60;
  *(_QWORD *)(__readfsqword(0) - 24) = &v38;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_C9FC90;
  if ( !&_pthread_key_create )
  {
    v13 = -1;
    goto LABEL_29;
  }
  v13 = pthread_once((pthread_once_t *)(v12 + 832), init_routine);
  if ( v13 )
    goto LABEL_29;
  if ( !qword_4F84F60 )
    sub_C7D570(&qword_4F84F60, (__int64 (*)(void))sub_CA0780, (__int64)sub_C9FD10);
  mutex = (pthread_mutex_t *)(qword_4F84F60 + 664);
  v13 = pthread_mutex_lock((pthread_mutex_t *)(qword_4F84F60 + 664));
  if ( v13 )
LABEL_29:
    sub_4264C5(v13);
  v14 = sub_C92610();
  v15 = (unsigned int)sub_C92740(v12 + 840, src, n, v14);
  v16 = *(_QWORD *)(v12 + 840);
  v17 = *(_QWORD *)(v16 + 8 * v15);
  if ( !v17 )
  {
LABEL_17:
    v30 = (_QWORD *)(v16 + 8 * v15);
    v31 = v15;
    v21 = sub_C7D670(n + 41, 8);
    v22 = v31;
    v23 = v30;
    v24 = (_QWORD *)v21;
    if ( n )
    {
      v29 = (_QWORD *)v21;
      memcpy((void *)(v21 + 40), src, n);
      v22 = v31;
      v23 = v30;
      v24 = v29;
    }
    *((_BYTE *)v24 + n + 40) = 0;
    *v24 = n;
    v24[1] = 0;
    v24[2] = 0;
    v24[3] = 0;
    v24[4] = 0xB800000000LL;
    *v23 = v24;
    ++*(_DWORD *)(v12 + 852);
    v25 = (__int64 *)(*(_QWORD *)(v12 + 840) + 8LL * (unsigned int)sub_C929D0((__int64 *)(v12 + 840), v22));
    v17 = *v25;
    if ( !*v25 || v17 == -8 )
    {
      v26 = v25 + 1;
      do
      {
        do
          v17 = *v26++;
        while ( v17 == -8 );
      }
      while ( !v17 );
    }
    goto LABEL_12;
  }
  if ( v17 == -8 )
  {
    --*(_DWORD *)(v12 + 856);
    goto LABEL_17;
  }
LABEL_12:
  if ( !*(_QWORD *)(v17 + 8) )
  {
    v33 = v17;
    v27 = (__int64 *)sub_22077B0(112);
    v17 = v33;
    v28 = v27;
    if ( v27 )
    {
      sub_C9E8E0(v27, src, n, a9, a10);
      v17 = v33;
    }
    *(_QWORD *)(v17 + 8) = v28;
  }
  v34 = v17;
  v18 = sub_C92610();
  v19 = *sub_CA07C0(v34 + 16, a2, a3, v18);
  v20 = v19 + 8;
  if ( !*(_QWORD *)(v19 + 160) )
    sub_C9EA20(v19 + 8, (__int64)a2, a3, a4, a5, *(_QWORD *)(v34 + 8));
  pthread_mutex_unlock(mutex);
  *a1 = v20;
  return sub_C9E250(v20);
}
