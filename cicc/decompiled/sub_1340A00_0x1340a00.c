// Function: sub_1340A00
// Address: 0x1340a00
//
_QWORD *__fastcall sub_1340A00(_BYTE *a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r12

  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 88)) )
  {
    sub_130AD90(a2 + 24);
    *(_BYTE *)(a2 + 128) = 1;
  }
  ++*(_QWORD *)(a2 + 80);
  if ( a1 != *(_BYTE **)(a2 + 72) )
  {
    ++*(_QWORD *)(a2 + 64);
    *(_QWORD *)(a2 + 72) = a1;
  }
  v2 = sub_133DFB0((_QWORD *)a2);
  v3 = v2;
  if ( v2 )
  {
    sub_133EB60((_QWORD *)a2, v2);
    --*(_QWORD *)(a2 + 16);
    *(_BYTE *)(a2 + 128) = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(a2 + 88));
    return v3;
  }
  else
  {
    *(_BYTE *)(a2 + 128) = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(a2 + 88));
    return sub_131C450(a1, *(_QWORD *)(a2 + 136));
  }
}
