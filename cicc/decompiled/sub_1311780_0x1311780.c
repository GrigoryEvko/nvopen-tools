// Function: sub_1311780
// Address: 0x1311780
//
int __fastcall sub_1311780(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rax
  __int64 v4; // rax
  _QWORD *v5; // rdx
  int result; // eax
  __int64 v7; // rdx

  v2 = *(_QWORD *)(a2 + 40);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(v2 + 10472)) )
  {
    sub_130AD90(v2 + 10408);
    *(_BYTE *)(v2 + 10512) = 1;
  }
  ++*(_QWORD *)(v2 + 10464);
  if ( a1 != *(_QWORD *)(v2 + 10456) )
  {
    ++*(_QWORD *)(v2 + 10448);
    *(_QWORD *)(v2 + 10456) = a1;
  }
  if ( a2 != *(_QWORD *)(v2 + 10392) )
    goto LABEL_6;
  if ( a2 != *(_QWORD *)a2 )
  {
    *(_QWORD *)(v2 + 10392) = *(_QWORD *)a2;
LABEL_6:
    **(_QWORD **)(a2 + 8) = *(_QWORD *)(*(_QWORD *)a2 + 8LL);
    v3 = *(_QWORD **)(a2 + 8);
    *(_QWORD *)(*(_QWORD *)a2 + 8LL) = v3;
    *(_QWORD *)(a2 + 8) = *v3;
    **(_QWORD **)(*(_QWORD *)a2 + 8LL) = *(_QWORD *)a2;
    **(_QWORD **)(a2 + 8) = a2;
    goto LABEL_7;
  }
  *(_QWORD *)(v2 + 10392) = 0;
LABEL_7:
  v4 = a2 + 16;
  if ( *(_QWORD *)(v2 + 10400) == a2 + 16 )
  {
    v7 = *(_QWORD *)(a2 + 16);
    if ( v4 == v7 )
    {
      *(_QWORD *)(v2 + 10400) = 0;
      goto LABEL_9;
    }
    *(_QWORD *)(v2 + 10400) = v7;
  }
  **(_QWORD **)(a2 + 24) = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL);
  v5 = *(_QWORD **)(a2 + 24);
  *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) = v5;
  *(_QWORD *)(a2 + 24) = *v5;
  **(_QWORD **)(*(_QWORD *)(a2 + 16) + 8LL) = *(_QWORD *)(a2 + 16);
  **(_QWORD **)(a2 + 24) = v4;
LABEL_9:
  sub_1311650(a1, *(_QWORD *)(a2 + 168), v2);
  *(_BYTE *)(v2 + 10512) = 0;
  result = pthread_mutex_unlock((pthread_mutex_t *)(v2 + 10472));
  *(_QWORD *)(a2 + 40) = 0;
  return result;
}
