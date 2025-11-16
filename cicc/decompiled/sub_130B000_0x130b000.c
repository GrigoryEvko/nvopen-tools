// Function: sub_130B000
// Address: 0x130b000
//
int __fastcall sub_130B000(__int64 a1, __int64 a2)
{
  int result; // eax

  result = pthread_mutex_trylock((pthread_mutex_t *)(a2 + 64));
  if ( result )
  {
    result = sub_130AD90(a2);
    *(_BYTE *)(a2 + 104) = 1;
  }
  ++*(_QWORD *)(a2 + 56);
  if ( a1 != *(_QWORD *)(a2 + 48) )
  {
    ++*(_QWORD *)(a2 + 40);
    *(_QWORD *)(a2 + 48) = a1;
  }
  return result;
}
