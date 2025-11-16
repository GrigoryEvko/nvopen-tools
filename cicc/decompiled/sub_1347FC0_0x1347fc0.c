// Function: sub_1347FC0
// Address: 0x1347fc0
//
_QWORD *__fastcall sub_1347FC0(_BYTE *a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  pthread_mutex_t *v4; // r14
  __int64 v7; // rsi
  _QWORD *v8; // r12
  void *v9; // r15
  _QWORD *v11; // rax
  __int64 v12; // rsi
  char v13[49]; // [rsp+Fh] [rbp-31h] BYREF

  v4 = (pthread_mutex_t *)(a2 + 176);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 176)) )
  {
    sub_130AD90(a2 + 112);
    *(_BYTE *)(a2 + 216) = 1;
  }
  ++*(_QWORD *)(a2 + 168);
  if ( a1 != *(_BYTE **)(a2 + 160) )
  {
    ++*(_QWORD *)(a2 + 152);
    *(_QWORD *)(a2 + 160) = a1;
  }
  *a4 = 0;
  if ( !*(_QWORD *)(a2 + 224) )
  {
    v13[0] = 1;
    v9 = (void *)sub_130CA40(0, 0x10000000u, 0x200000, v13);
    if ( v9 )
    {
      v8 = sub_131C440(a1, *(_QWORD *)(a2 + 240), 248, 64);
      if ( v8 )
      {
        *(_QWORD *)(a2 + 224) = v9;
        *(_QWORD *)(a2 + 232) = 0x10000000;
        goto LABEL_9;
      }
      sub_130CB40(v9, 0x10000000u);
    }
    *a4 = 1;
    v8 = 0;
    *(_BYTE *)(a2 + 216) = 0;
    pthread_mutex_unlock(v4);
    return v8;
  }
  v7 = *(_QWORD *)(a2 + 240);
  if ( *(_QWORD *)(a2 + 232) == 0x200000 )
  {
    v11 = sub_131C440(a1, v7, 248, 64);
    v8 = v11;
    if ( v11 )
    {
      v12 = *(_QWORD *)(a2 + 224);
      ++*(_QWORD *)(a2 + 248);
      sub_1349790(v11, v12);
      *(_QWORD *)(a2 + 224) = 0;
      *(_QWORD *)(a2 + 232) = 0;
      *(_BYTE *)(a2 + 216) = 0;
      pthread_mutex_unlock(v4);
      return v8;
    }
LABEL_16:
    *a4 = 1;
    *(_BYTE *)(a2 + 216) = 0;
    pthread_mutex_unlock(v4);
    return v8;
  }
  v8 = sub_131C440(a1, v7, 248, 64);
  if ( !v8 )
    goto LABEL_16;
  v9 = *(void **)(a2 + 224);
LABEL_9:
  ++*(_QWORD *)(a2 + 248);
  sub_1349790(v8, v9);
  *(_QWORD *)(a2 + 224) += 0x200000LL;
  *(_QWORD *)(a2 + 232) -= 0x200000LL;
  *(_BYTE *)(a2 + 216) = 0;
  pthread_mutex_unlock(v4);
  return v8;
}
