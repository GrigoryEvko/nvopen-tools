// Function: sub_1340D90
// Address: 0x1340d90
//
int __fastcall sub_1340D90(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v4; // rsi
  __int64 i; // r12
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // rdi
  int result; // eax

  v2 = *(_QWORD *)(a2 + 8);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(v2 + 88)) )
  {
    sub_130AD90(v2 + 24);
    *(_BYTE *)(v2 + 128) = 1;
  }
  ++*(_QWORD *)(v2 + 80);
  if ( a1 != *(_QWORD *)(v2 + 72) )
  {
    ++*(_QWORD *)(v2 + 64);
    *(_QWORD *)(v2 + 72) = a1;
  }
  v4 = *(_QWORD **)a2;
  for ( i = 0; *(_QWORD *)a2; v4 = *(_QWORD **)a2 )
  {
    v7 = (_QWORD *)v4[8];
    *(_QWORD *)a2 = v7;
    if ( v7 == v4 )
    {
      *(_QWORD *)a2 = 0;
    }
    else
    {
      *(_QWORD *)(v4[9] + 64LL) = v7[9];
      v6 = v4[9];
      *(_QWORD *)(v4[8] + 72LL) = v6;
      v4[9] = *(_QWORD *)(v6 + 64);
      *(_QWORD *)(*(_QWORD *)(v4[8] + 72LL) + 64LL) = v4[8];
      *(_QWORD *)(v4[9] + 64LL) = v4;
    }
    ++i;
    sub_133E330(*(_QWORD **)(a2 + 8), v4);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 8) + 16LL) += i;
  v8 = *(_QWORD *)(a2 + 8);
  *(_BYTE *)(v8 + 128) = 0;
  result = pthread_mutex_unlock((pthread_mutex_t *)(v8 + 88));
  *(_BYTE *)(a2 + 16) = 1;
  return result;
}
