// Function: sub_130EDA0
// Address: 0x130eda0
//
_QWORD *__fastcall sub_130EDA0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // r12
  unsigned __int64 v4; // r14
  __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // r15

  v3 = *(_QWORD *)(a2 + 64);
  if ( v3 )
  {
    v4 = 0;
    v3 = 0;
    do
    {
      v6 = 144 * v4 + *(_QWORD *)(a2 + 104);
      v7 = 144 * v4;
      if ( pthread_mutex_trylock((pthread_mutex_t *)(v6 + 64)) )
      {
        sub_130AD90(v6);
        *(_BYTE *)(v6 + 104) = 1;
      }
      ++*(_QWORD *)(v6 + 56);
      if ( a1 != *(_QWORD *)(v6 + 48) )
      {
        ++*(_QWORD *)(v6 + 40);
        *(_QWORD *)(v6 + 48) = a1;
      }
      ++v4;
      v5 = v7 + *(_QWORD *)(a2 + 104);
      v3 += *(_QWORD *)(v5 + 128);
      v5 += 64;
      *(_BYTE *)(v5 + 40) = 0;
      pthread_mutex_unlock((pthread_mutex_t *)v5);
    }
    while ( *(_QWORD *)(a2 + 64) > v4 );
  }
  *a3 += v3;
  return a3;
}
