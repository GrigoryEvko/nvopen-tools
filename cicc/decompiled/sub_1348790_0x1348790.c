// Function: sub_1348790
// Address: 0x1348790
//
int __fastcall sub_1348790(__int64 a1, __int64 a2)
{
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
  sub_1340D90(a1, a2 + 296);
  *(_BYTE *)(a2 + 168) = 0;
  return pthread_mutex_unlock((pthread_mutex_t *)(a2 + 128));
}
