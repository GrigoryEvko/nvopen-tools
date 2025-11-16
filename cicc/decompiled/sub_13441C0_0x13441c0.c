// Function: sub_13441C0
// Address: 0x13441c0
//
unsigned __int64 *__fastcall sub_13441C0(_BYTE *a1, __int64 a2, unsigned int *a3, __int64 a4, unsigned __int64 a5)
{
  unsigned __int64 *v6; // r15
  __int64 v7; // r9
  __int64 v8; // rbx
  __int64 *v9; // rbx
  __int64 v10; // rsi
  pthread_mutex_t *mutex; // [rsp+0h] [rbp-70h]
  __int64 v15; // [rsp+20h] [rbp-50h]
  __int64 i; // [rsp+28h] [rbp-48h]
  _BYTE v17[49]; // [rsp+3Fh] [rbp-31h] BYREF

  mutex = (pthread_mutex_t *)(a4 + 64);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a4 + 64)) )
  {
    sub_130AD90(a4);
    *(_BYTE *)(a4 + 104) = 1;
  }
  ++*(_QWORD *)(a4 + 56);
  if ( a1 != *(_BYTE **)(a4 + 48) )
  {
    ++*(_QWORD *)(a4 + 40);
    *(_QWORD *)(a4 + 48) = a1;
  }
  for ( i = a4 + 112; ; sub_1342830(i, v9) )
  {
    v6 = *(unsigned __int64 **)(a4 + 9744);
    v7 = a4 + 112;
    if ( !v6 )
    {
      v6 = *(unsigned __int64 **)(a4 + 19400);
      if ( !v6 )
      {
LABEL_18:
        v6 = 0;
        goto LABEL_13;
      }
      v7 = a4 + 9768;
    }
    v15 = v7;
    v8 = sub_13427E0(i);
    if ( a5 >= sub_13427E0(a4 + 9768) + v8 )
      goto LABEL_18;
    sub_1342A40(v15, v6);
    if ( !*(_BYTE *)(a4 + 19432) )
      break;
    if ( (*v6 & 0x10000) != 0 )
      break;
    sub_1341570((__int64)a1, *(_QWORD *)(a2 + 58384), v6, 0);
    v9 = sub_13436B0(a1, a2, a3, a4, (__int64 *)v6, v17);
    sub_1341570((__int64)a1, *(_QWORD *)(a2 + 58384), (unsigned __int64 *)v9, *(_DWORD *)(a4 + 19424));
    if ( !v17[0] )
      break;
  }
  v10 = *(_QWORD *)(a2 + 58384);
  if ( *(_DWORD *)(a4 + 19424) <= 2u )
    sub_1341570((__int64)a1, v10, v6, 0);
  else
    sub_1341E90((__int64)a1, v10, (__int64)v6);
LABEL_13:
  *(_BYTE *)(a4 + 104) = 0;
  pthread_mutex_unlock(mutex);
  return v6;
}
