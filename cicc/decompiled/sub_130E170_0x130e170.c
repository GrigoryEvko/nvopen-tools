// Function: sub_130E170
// Address: 0x130e170
//
__int64 __fastcall sub_130E170(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // r9
  __int64 v4; // r8
  unsigned __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rcx
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v15; // rax
  char v16; // [rsp+7h] [rbp-19h] BYREF
  _QWORD v17[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = 0;
  v4 = 0;
  v17[0] = 0;
LABEL_2:
  v7 = *(_QWORD *)(a3 + 128);
  while ( v7 > *(_QWORD *)(a2 + 88) )
  {
    v8 = 3LL * *(unsigned int *)(a3 + 136);
    v9 = *(_QWORD *)(a3 + 120);
    v10 = *(_DWORD *)(a3 + 136) + 1;
    *(_DWORD *)(a3 + 136) = v10;
    v11 = v9 + 8 * v8;
    if ( v10 == *(_DWORD *)(a2 + 112) )
      *(_DWORD *)(a3 + 136) = 0;
    v12 = *(_QWORD *)(v11 + 8);
    if ( v12 )
    {
      *(_QWORD *)(a3 + 128) = v7 - v12;
      *(_QWORD *)(v11 + 8) = 0;
      if ( v4 )
      {
        v15 = *(_QWORD *)(v11 + 16);
        if ( v15 )
        {
          *(_QWORD *)(*(_QWORD *)(v15 + 48) + 40LL) = *(_QWORD *)(v4 + 48);
          *(_QWORD *)(v4 + 48) = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 48LL);
          *(_QWORD *)(*(_QWORD *)(v11 + 16) + 48LL) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 16) + 48LL) + 40LL);
          *(_QWORD *)(*(_QWORD *)(v4 + 48) + 40LL) = v4;
          *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v11 + 16) + 48LL) + 40LL) = *(_QWORD *)(v11 + 16);
          *(_QWORD *)(v11 + 16) = 0;
        }
      }
      else
      {
        v4 = *(_QWORD *)(v11 + 16);
        v3 = 1;
        *(_QWORD *)(v11 + 16) = 0;
      }
      goto LABEL_2;
    }
  }
  if ( v3 )
    v17[0] = v4;
  *(_BYTE *)(a3 + 104) = 0;
  pthread_mutex_unlock((pthread_mutex_t *)(a3 + 64));
  v13 = *(_QWORD *)(a2 + 56);
  v16 = 0;
  return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, char *))(v13 + 40))(a1, v13, v17, &v16);
}
