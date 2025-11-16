// Function: sub_1314710
// Address: 0x1314710
//
int __fastcall sub_1314710(__int64 a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  int result; // eax
  __int64 v8; // rcx
  int v9; // edx
  __int64 *v10; // r15
  __int64 v11; // rdi
  __int64 v12; // rbx

  result = pthread_mutex_trylock((pthread_mutex_t *)(a2 + 64));
  if ( result )
  {
    *(_BYTE *)(a2 + 104) = 1;
    v8 = (unsigned int)*a4;
    a3[v8] = a2;
    v9 = v8 + 1;
    if ( (_DWORD)v8 == 31 )
    {
      v10 = a3 + 32;
      do
      {
        v12 = *a3;
        if ( pthread_mutex_trylock((pthread_mutex_t *)(*a3 + 64)) )
        {
          sub_130AD90(v12);
          *(_BYTE *)(v12 + 104) = 1;
        }
        ++*(_QWORD *)(v12 + 56);
        if ( a1 != *(_QWORD *)(v12 + 48) )
        {
          ++*(_QWORD *)(v12 + 40);
          *(_QWORD *)(v12 + 48) = a1;
        }
        v11 = *a3++;
        *(_BYTE *)(v11 + 104) = 0;
        result = pthread_mutex_unlock((pthread_mutex_t *)(v11 + 64));
      }
      while ( v10 != a3 );
      v9 = 0;
    }
    *a4 = v9;
  }
  else
  {
    ++*(_QWORD *)(a2 + 56);
    if ( a1 != *(_QWORD *)(a2 + 48) )
    {
      ++*(_QWORD *)(a2 + 40);
      *(_QWORD *)(a2 + 48) = a1;
    }
    *(_BYTE *)(a2 + 104) = 0;
    return pthread_mutex_unlock((pthread_mutex_t *)(a2 + 64));
  }
  return result;
}
