// Function: sub_131C490
// Address: 0x131c490
//
int __fastcall sub_131C490(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4, _QWORD *a5, _QWORD *a6)
{
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 96)) )
  {
    sub_130AD90(a2 + 32);
    *(_BYTE *)(a2 + 136) = 1;
  }
  ++*(_QWORD *)(a2 + 88);
  if ( a1 != *(_QWORD *)(a2 + 80) )
  {
    ++*(_QWORD *)(a2 + 72);
    *(_QWORD *)(a2 + 80) = a1;
  }
  *a3 = *(_QWORD *)(a2 + 3880);
  *a4 = *(_QWORD *)(a2 + 3888);
  *a5 = *(_QWORD *)(a2 + 3896);
  *a6 = *(_QWORD *)(a2 + 3904);
  *(_BYTE *)(a2 + 136) = 0;
  return pthread_mutex_unlock((pthread_mutex_t *)(a2 + 96));
}
