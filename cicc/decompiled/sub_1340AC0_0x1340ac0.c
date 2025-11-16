// Function: sub_1340AC0
// Address: 0x1340ac0
//
int __fastcall sub_1340AC0(__int64 a1, __int64 a2, _QWORD *a3)
{
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 88)) )
  {
    sub_130AD90(a2 + 24);
    *(_BYTE *)(a2 + 128) = 1;
  }
  ++*(_QWORD *)(a2 + 80);
  if ( a1 != *(_QWORD *)(a2 + 72) )
  {
    ++*(_QWORD *)(a2 + 64);
    *(_QWORD *)(a2 + 72) = a1;
  }
  sub_133E330((_QWORD *)a2, a3);
  ++*(_QWORD *)(a2 + 16);
  *(_BYTE *)(a2 + 128) = 0;
  return pthread_mutex_unlock((pthread_mutex_t *)(a2 + 88));
}
