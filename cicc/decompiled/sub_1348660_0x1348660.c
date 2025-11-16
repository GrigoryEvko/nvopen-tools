// Function: sub_1348660
// Address: 0x1348660
//
int __fastcall sub_1348660(__int64 a1, __int64 a2, _QWORD *a3)
{
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 240)) )
  {
    sub_130AD90(a2 + 176);
    *(_BYTE *)(a2 + 280) = 1;
  }
  ++*(_QWORD *)(a2 + 232);
  if ( a1 != *(_QWORD *)(a2 + 224) )
  {
    ++*(_QWORD *)(a2 + 216);
    *(_QWORD *)(a2 + 224) = a1;
  }
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 128)) )
  {
    sub_130AD90(a2 + 64);
    *(_BYTE *)(a2 + 168) = 1;
  }
  ++*(_QWORD *)(a2 + 120);
  if ( a1 != *(_QWORD *)(a2 + 112) )
  {
    ++*(_QWORD *)(a2 + 104);
    *(_QWORD *)(a2 + 112) = a1;
  }
  sub_134BBB0(a3, a2 + 1376);
  a3[396] += *(_QWORD *)(a2 + 5672);
  a3[397] += *(_QWORD *)(a2 + 5680);
  a3[398] += *(_QWORD *)(a2 + 5688);
  a3[399] += *(_QWORD *)(a2 + 5696);
  *(_BYTE *)(a2 + 168) = 0;
  pthread_mutex_unlock((pthread_mutex_t *)(a2 + 128));
  *(_BYTE *)(a2 + 280) = 0;
  return pthread_mutex_unlock((pthread_mutex_t *)(a2 + 240));
}
