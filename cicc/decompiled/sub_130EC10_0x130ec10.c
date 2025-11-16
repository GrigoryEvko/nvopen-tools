// Function: sub_130EC10
// Address: 0x130ec10
//
void __fastcall sub_130EC10(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r15
  __int64 v3; // rbx
  __int64 v4; // rbx
  __int64 v5; // r14

  if ( *(_QWORD *)(a2 + 64) )
  {
    v2 = 0;
    do
    {
      v4 = 144 * v2;
      v5 = 144 * v2 + *(_QWORD *)(a2 + 104);
      if ( pthread_mutex_trylock((pthread_mutex_t *)(v5 + 64)) )
      {
        sub_130AD90(v5);
        *(_BYTE *)(v5 + 104) = 1;
      }
      ++*(_QWORD *)(v5 + 56);
      if ( a1 != *(_QWORD *)(v5 + 48) )
      {
        ++*(_QWORD *)(v5 + 40);
        *(_QWORD *)(v5 + 48) = a1;
      }
      ++v2;
      sub_130E080(a1, a2, v4 + *(_QWORD *)(a2 + 104));
      v3 = *(_QWORD *)(a2 + 104) + v4;
      *(_BYTE *)(v3 + 104) = 0;
      pthread_mutex_unlock((pthread_mutex_t *)(v3 + 64));
    }
    while ( *(_QWORD *)(a2 + 64) > v2 );
  }
}
