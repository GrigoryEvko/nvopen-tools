// Function: sub_13481F0
// Address: 0x13481f0
//
__int64 __fastcall sub_13481F0(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, bool *a6)
{
  __int64 v8; // r14
  _QWORD *v10; // rax
  _QWORD *v11; // rsi
  __int64 v12; // rax
  _QWORD *v13; // [rsp+18h] [rbp-70h]
  pthread_mutex_t *mutex; // [rsp+28h] [rbp-60h]
  _BYTE v18[49]; // [rsp+57h] [rbp-31h] BYREF

  v18[0] = 0;
  v8 = sub_1347C30(a1, a2, a3, v18, a4, a5, a6);
  if ( a4 != v8 && !v18[0] )
  {
    mutex = (pthread_mutex_t *)(a2 + 240);
    if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 240)) )
    {
      sub_130AD90(a2 + 176);
      *(_BYTE *)(a2 + 280) = 1;
    }
    ++*(_QWORD *)(a2 + 232);
    if ( a1 != *(_BYTE **)(a2 + 224) )
    {
      ++*(_QWORD *)(a2 + 216);
      *(_QWORD *)(a2 + 224) = a1;
    }
    v8 += sub_1347C30(a1, a2, a3, v18, a4 - v8, a5, a6);
    if ( a4 == v8 || v18[0] || (v10 = sub_1347FC0(a1, *(_QWORD *)(a2 + 56), a3, v18)) == 0 )
    {
      *(_BYTE *)(a2 + 280) = 0;
      pthread_mutex_unlock(mutex);
    }
    else
    {
      v13 = v10;
      v11 = v10;
      if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 128)) )
      {
        sub_130AD90(a2 + 64);
        *(_BYTE *)(a2 + 168) = 1;
        v11 = v13;
      }
      ++*(_QWORD *)(a2 + 120);
      if ( a1 != *(_BYTE **)(a2 + 112) )
      {
        ++*(_QWORD *)(a2 + 104);
        *(_QWORD *)(a2 + 112) = a1;
      }
      sub_134BFB0(a2 + 320, v11);
      *(_BYTE *)(a2 + 168) = 0;
      pthread_mutex_unlock((pthread_mutex_t *)(a2 + 128));
      v12 = sub_1347C30(a1, a2, a3, v18, a4 - v8, a5, a6);
      *(_BYTE *)(a2 + 280) = 0;
      v8 += v12;
      pthread_mutex_unlock(mutex);
    }
  }
  return v8;
}
