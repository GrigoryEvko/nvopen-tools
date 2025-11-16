// Function: sub_1348810
// Address: 0x1348810
//
int __fastcall sub_1348810(__int64 a1, __int64 a2, char a3)
{
  char v4; // al

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
  v4 = *(_BYTE *)(a2 + 5644);
  *(_BYTE *)(a2 + 5644) = a3;
  if ( a3 != 1 && v4 )
    sub_1347550(a1, a2, 1);
  *(_BYTE *)(a2 + 168) = 0;
  return pthread_mutex_unlock((pthread_mutex_t *)(a2 + 128));
}
