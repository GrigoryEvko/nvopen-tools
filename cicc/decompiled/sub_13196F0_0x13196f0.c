// Function: sub_13196F0
// Address: 0x13196f0
//
__int64 __fastcall sub_13196F0(_BYTE *a1, __int64 a2)
{
  unsigned int v2; // r13d
  char v3; // al
  char v5; // al
  void *thread_return; // [rsp+8h] [rbp-28h] BYREF

  ++a1[1];
  if ( !a1[816] )
    sub_1313A40(a1);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 120)) )
  {
    sub_130AD90(a2 + 56);
    *(_BYTE *)(a2 + 160) = 1;
  }
  ++*(_QWORD *)(a2 + 112);
  if ( a1 != *(_BYTE **)(a2 + 104) )
  {
    ++*(_QWORD *)(a2 + 96);
    *(_QWORD *)(a2 + 104) = a1;
  }
  if ( *(_DWORD *)(a2 + 168) == 1 )
  {
    *(_DWORD *)(a2 + 168) = 0;
    pthread_cond_signal((pthread_cond_t *)(a2 + 8));
    *(_BYTE *)(a2 + 160) = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(a2 + 120));
    if ( pthread_join(*(_QWORD *)a2, &thread_return) )
    {
      v2 = 1;
      v5 = a1[1] - 1;
      a1[1] = v5;
      if ( v5 )
        return v2;
      goto LABEL_14;
    }
    --unk_5260D40;
  }
  else
  {
    *(_BYTE *)(a2 + 160) = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(a2 + 120));
  }
  v2 = 0;
  v3 = a1[1] - 1;
  a1[1] = v3;
  if ( v3 )
    return v2;
LABEL_14:
  sub_1313A40(a1);
  return v2;
}
