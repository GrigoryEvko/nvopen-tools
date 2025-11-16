// Function: sub_1347400
// Address: 0x1347400
//
unsigned __int64 __fastcall sub_1347400(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 128)) )
  {
    sub_130AD90(a2 + 64);
    *(_BYTE *)(a2 + 168) = 1;
  }
  ++*(_QWORD *)(a2 + 120);
  if ( a1 != *(_QWORD *)(a2 + 112) )
  {
    ++*(_QWORD *)(a2 + 104);
    *(_QWORD *)(a2 + 112) = a1;
  }
  v2 = sub_134BFA0(a2 + 320);
  if ( v2 )
  {
    v10[0] = *(_QWORD *)(v2 + 24);
    v3 = (*(__int64 (__fastcall **)(_QWORD *))(*(_QWORD *)(a2 + 56) + 304LL))(v10);
    v4 = *(_QWORD *)(a2 + 5648);
    if ( v4 <= v3 )
      goto LABEL_14;
    v5 = 1000000 * (v4 - v3);
  }
  else
  {
    v5 = -1;
  }
  if ( !sub_1347320(a2) )
  {
LABEL_13:
    *(_BYTE *)(a2 + 168) = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(a2 + 128));
    return v5;
  }
  if ( *(_QWORD *)(a2 + 5672) )
  {
    v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)(a2 + 56) + 304LL))(a2 + 5704);
    v7 = *(_QWORD *)(a2 + 5656);
    if ( v7 <= v6 )
    {
      v5 = 0;
    }
    else
    {
      v8 = 1000000 * (v7 - v6);
      if ( v5 > v8 )
        v5 = v8;
    }
    goto LABEL_13;
  }
LABEL_14:
  *(_BYTE *)(a2 + 168) = 0;
  pthread_mutex_unlock((pthread_mutex_t *)(a2 + 128));
  return 0;
}
