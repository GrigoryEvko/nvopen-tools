// Function: sub_1316E90
// Address: 0x1316e90
//
__int64 __fastcall sub_1316E90(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r12

  v4 = unk_5260DD8 + 208 * ((unsigned __int64)*(unsigned int *)(a2 + 78928) % unk_5260D48);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(v4 + 120)) )
  {
    sub_130AD90(v4 + 56);
    *(_BYTE *)(v4 + 160) = 1;
  }
  ++*(_QWORD *)(v4 + 112);
  if ( a1 != *(_QWORD *)(v4 + 104) )
  {
    ++*(_QWORD *)(v4 + 96);
    *(_QWORD *)(v4 + 104) = a1;
  }
  sub_130B460(a1, a2 + 10648);
  v5 = sub_131C420(*(_QWORD *)(a2 + 78936), a3);
  *(_BYTE *)(v4 + 160) = 0;
  v6 = v5;
  pthread_mutex_unlock((pthread_mutex_t *)(v4 + 120));
  return v6;
}
