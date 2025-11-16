// Function: sub_13094D0
// Address: 0x13094d0
//
__int64 __fastcall sub_13094D0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned int v5; // eax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax

  v5 = *(_DWORD *)(a2 + 78928);
  if ( !a4 )
  {
    if ( unk_5057900 > v5 )
      return sub_1314E00(a1, a2, a3);
    if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 10600)) )
    {
      sub_130AD90(a2 + 10536);
      *(_BYTE *)(a2 + 10640) = 1;
    }
    ++*(_QWORD *)(a2 + 10592);
    if ( a1 != *(_QWORD *)(a2 + 10584) )
    {
      ++*(_QWORD *)(a2 + 10576);
      *(_QWORD *)(a2 + 10584) = a1;
    }
    if ( a3 == *(_QWORD *)(a2 + 10528) )
    {
      v10 = *(_QWORD *)(a3 + 40);
      if ( a3 == v10 )
      {
        *(_QWORD *)(a2 + 10528) = 0;
        goto LABEL_13;
      }
      *(_QWORD *)(a2 + 10528) = v10;
    }
    *(_QWORD *)(*(_QWORD *)(a3 + 48) + 40LL) = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL);
    v8 = *(_QWORD *)(a3 + 48);
    *(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL) = v8;
    *(_QWORD *)(a3 + 48) = *(_QWORD *)(v8 + 40);
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL) + 40LL) = *(_QWORD *)(a3 + 40);
    *(_QWORD *)(*(_QWORD *)(a3 + 48) + 40LL) = a3;
LABEL_13:
    *(_BYTE *)(a2 + 10640) = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(a2 + 10600));
    return sub_1314E00(a1, a2, a3);
  }
  if ( unk_5057900 <= v5 )
  {
    if ( a3 == *(_QWORD *)(a2 + 10528) )
    {
      v9 = *(_QWORD *)(a3 + 40);
      if ( a3 == v9 )
      {
        *(_QWORD *)(a2 + 10528) = 0;
        return sub_1314E00(a1, a2, a3);
      }
      *(_QWORD *)(a2 + 10528) = v9;
    }
    *(_QWORD *)(*(_QWORD *)(a3 + 48) + 40LL) = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL);
    v7 = *(_QWORD *)(a3 + 48);
    *(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL) = v7;
    *(_QWORD *)(a3 + 48) = *(_QWORD *)(v7 + 40);
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL) + 40LL) = *(_QWORD *)(a3 + 40);
    *(_QWORD *)(*(_QWORD *)(a3 + 48) + 40LL) = a3;
  }
  return sub_1314E00(a1, a2, a3);
}
