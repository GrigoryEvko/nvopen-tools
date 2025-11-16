// Function: sub_130BC50
// Address: 0x130bc50
//
unsigned __int64 __fastcall sub_130BC50(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 v4; // rax
  unsigned __int64 v5; // r13
  __int64 v6; // r15
  __int64 v7; // rax
  unsigned __int64 v8; // r12
  __int64 v10; // [rsp+8h] [rbp-38h]

  v2 = sub_13427E0(a2 + 168);
  v3 = sub_13427E0(a2 + 9824);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 58712)) )
  {
    *(_BYTE *)(a2 + 58752) = 1;
    return 0;
  }
  else
  {
    ++*(_QWORD *)(a2 + 58704);
    if ( a1 != *(_QWORD *)(a2 + 58696) )
    {
      ++*(_QWORD *)(a2 + 58688);
      *(_QWORD *)(a2 + 58696) = a1;
    }
    v4 = sub_133DC20(a2 + 58648, v2 + v3, 1024);
    *(_BYTE *)(a2 + 58752) = 0;
    v5 = v4;
    pthread_mutex_unlock((pthread_mutex_t *)(a2 + 58712));
    if ( v5 )
    {
      v10 = sub_13427E0(a2 + 19608);
      v6 = sub_13427E0(a2 + 29264);
      if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 60496)) )
      {
        *(_BYTE *)(a2 + 60536) = 1;
        return 0;
      }
      else
      {
        ++*(_QWORD *)(a2 + 60488);
        if ( a1 != *(_QWORD *)(a2 + 60480) )
        {
          ++*(_QWORD *)(a2 + 60472);
          *(_QWORD *)(a2 + 60480) = a1;
        }
        v7 = sub_133DC20(a2 + 60432, v6 + v10, 1024);
        *(_BYTE *)(a2 + 60536) = 0;
        v8 = v7;
        pthread_mutex_unlock((pthread_mutex_t *)(a2 + 60496));
        if ( v5 > v8 )
          return v8;
      }
    }
  }
  return v5;
}
