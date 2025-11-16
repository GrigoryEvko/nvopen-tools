// Function: sub_133A200
// Address: 0x133a200
//
int __fastcall sub_133A200(__int64 a1, unsigned int a2)
{
  int result; // eax
  unsigned __int64 v3; // rbx

  if ( pthread_mutex_trylock(&stru_5260DA0) )
  {
    sub_130AD90((__int64)qword_5260D60);
    unk_5260DC8 = 1;
  }
  ++unk_5260D98;
  if ( a1 != unk_5260D90 )
  {
    ++unk_5260D88;
    unk_5260D90 = a1;
  }
  result = byte_5260DD0[0];
  if ( byte_5260DD0[0] )
  {
    v3 = qword_5260DD8 + 208 * ((unsigned __int64)a2 % qword_5260D48[0]);
    if ( pthread_mutex_trylock((pthread_mutex_t *)(v3 + 120)) )
    {
      sub_130AD90(v3 + 56);
      *(_BYTE *)(v3 + 160) = 1;
    }
    ++*(_QWORD *)(v3 + 112);
    if ( a1 != *(_QWORD *)(v3 + 104) )
    {
      ++*(_QWORD *)(v3 + 96);
      *(_QWORD *)(v3 + 104) = a1;
    }
    *(_DWORD *)(v3 + 168) = 2;
    *(_BYTE *)(v3 + 160) = 0;
    return pthread_mutex_unlock((pthread_mutex_t *)(v3 + 120));
  }
  return result;
}
