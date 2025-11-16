// Function: sub_1340BA0
// Address: 0x1340ba0
//
_QWORD *__fastcall sub_1340BA0(_BYTE *a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // r13
  int v7; // r13d
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi

  if ( *(_BYTE *)(a2 + 16) )
    return sub_1340A00(a1, *(_QWORD *)(a2 + 8));
  result = *(_QWORD **)a2;
  if ( !*(_QWORD *)a2 )
  {
    v6 = *(_QWORD *)(a2 + 8);
    if ( pthread_mutex_trylock((pthread_mutex_t *)(v6 + 88)) )
    {
      sub_130AD90(v6 + 24);
      *(_BYTE *)(v6 + 128) = 1;
    }
    ++*(_QWORD *)(v6 + 80);
    if ( a1 != *(_BYTE **)(v6 + 72) )
    {
      ++*(_QWORD *)(v6 + 64);
      *(_QWORD *)(v6 + 72) = a1;
    }
    v7 = 4;
    do
    {
      v8 = sub_133E530(*(_QWORD **)(a2 + 8));
      if ( !v8 )
        break;
      v8[8] = v8;
      v8[9] = v8;
      if ( *(_QWORD *)a2 )
      {
        v8[8] = *(_QWORD *)(*(_QWORD *)a2 + 72LL);
        *(_QWORD *)(*(_QWORD *)a2 + 72LL) = v8;
        v8[9] = *(_QWORD *)(v8[9] + 64LL);
        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 72LL) + 64LL) = *(_QWORD *)a2;
        *(_QWORD *)(v8[9] + 64LL) = v8;
        v8 = (_QWORD *)v8[8];
      }
      v9 = *(_QWORD *)(a2 + 8);
      *(_QWORD *)a2 = v8;
      --*(_QWORD *)(v9 + 16);
      --v7;
    }
    while ( v7 );
    v10 = *(_QWORD *)(a2 + 8);
    *(_BYTE *)(v10 + 128) = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(v10 + 88));
    result = *(_QWORD **)a2;
    if ( !*(_QWORD *)a2 )
      return sub_131C450(a1, *(_QWORD *)(*(_QWORD *)(a2 + 8) + 136LL));
  }
  v4 = (_QWORD *)result[8];
  *(_QWORD *)a2 = v4;
  if ( result == v4 )
  {
    *(_QWORD *)a2 = 0;
  }
  else
  {
    *(_QWORD *)(result[9] + 64LL) = v4[9];
    v5 = result[9];
    *(_QWORD *)(result[8] + 72LL) = v5;
    result[9] = *(_QWORD *)(v5 + 64);
    *(_QWORD *)(*(_QWORD *)(result[8] + 72LL) + 64LL) = result[8];
    *(_QWORD *)(result[9] + 64LL) = result;
  }
  return result;
}
