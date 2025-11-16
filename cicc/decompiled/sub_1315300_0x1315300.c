// Function: sub_1315300
// Address: 0x1315300
//
__int64 __fastcall sub_1315300(__int64 a1, __int64 a2)
{
  pthread_mutex_t *v3; // r13
  _QWORD *v5; // r14
  _QWORD *v6; // rbx
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 *v9; // rax
  unsigned __int64 v10; // r8
  __int64 v11; // rbx
  pthread_mutex_t *v12; // r13
  unsigned __int64 *v13; // r14
  __int64 v14; // rdx
  unsigned __int64 *v15; // r14
  unsigned __int64 *v16; // rax
  unsigned __int64 v18; // rcx
  _QWORD *v19; // rcx
  unsigned int i; // edi
  __int64 v21; // r9
  _QWORD *v22; // rdi
  _QWORD *v23; // r9
  unsigned __int64 v24; // rax
  unsigned int *v25; // [rsp+8h] [rbp-1D8h]
  __int64 v26; // [rsp+10h] [rbp-1D0h]
  char *v27; // [rsp+10h] [rbp-1D0h]
  _BYTE *v28; // [rsp+18h] [rbp-1C8h]
  unsigned int j; // [rsp+18h] [rbp-1C8h]
  __int64 v30; // [rsp+20h] [rbp-1C0h]
  unsigned __int64 *v31; // [rsp+20h] [rbp-1C0h]
  _BYTE *v32; // [rsp+28h] [rbp-1B8h]
  _QWORD v33[54]; // [rsp+30h] [rbp-1B0h] BYREF

  v3 = (pthread_mutex_t *)(a2 + 10600);
  v26 = a2 + 10536;
  v28 = (_BYTE *)(a2 + 10640);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 10600)) )
  {
    sub_130AD90(v26);
    *(_BYTE *)(a2 + 10640) = 1;
  }
  ++*(_QWORD *)(a2 + 10592);
  if ( a1 != *(_QWORD *)(a2 + 10584) )
  {
    ++*(_QWORD *)(a2 + 10576);
    *(_QWORD *)(a2 + 10584) = a1;
  }
  v5 = *(_QWORD **)(a2 + 10528);
  if ( v5 )
  {
    v6 = (_QWORD *)(a1 + 432);
    if ( !a1 )
      v6 = v33;
    do
    {
      v30 = v5[1];
      *v28 = 0;
      pthread_mutex_unlock(v3);
      v7 = v30 & 0xFFFFFFFFFFFFF000LL;
      if ( !a1 )
      {
        sub_130D500(v33);
        v7 = v30 & 0xFFFFFFFFFFFFF000LL;
      }
      v8 = v30 & 0xFFFFFFFFC0000000LL;
      v9 = (_QWORD *)((char *)v6 + ((v7 >> 26) & 0xF0));
      v10 = *v9;
      if ( (v30 & 0xFFFFFFFFC0000000LL) != *v9 )
      {
        if ( v8 == v6[32] )
        {
          v6[32] = v10;
          v18 = v6[33];
          v6[33] = v9[1];
LABEL_46:
          *v9 = v8;
          v9[1] = v18;
        }
        else
        {
          v19 = v6 + 34;
          for ( i = 1; i != 8; ++i )
          {
            if ( v8 == *v19 )
            {
              v21 = 2LL * i;
              v22 = &v6[2 * i - 2];
              v23 = &v6[v21];
              v18 = v23[33];
              v23[32] = v22[32];
              v23[33] = v22[33];
              v22[32] = v10;
              v22[33] = v9[1];
              goto LABEL_46;
            }
            v19 += 2;
          }
          sub_130D370(a1, (__int64)&unk_5060AE0, v6, v7, 1, 0);
        }
      }
      sub_130A160(a1, v5);
      if ( pthread_mutex_trylock(v3) )
      {
        sub_130AD90(v26);
        *v28 = 1;
      }
      ++*(_QWORD *)(a2 + 10592);
      if ( a1 != *(_QWORD *)(a2 + 10584) )
      {
        ++*(_QWORD *)(a2 + 10576);
        *(_QWORD *)(a2 + 10584) = a1;
      }
      v5 = *(_QWORD **)(a2 + 10528);
    }
    while ( v5 );
  }
  *(_BYTE *)(a2 + 10640) = 0;
  pthread_mutex_unlock(v3);
  v27 = (char *)&unk_5260DF4;
  v25 = dword_5060A40;
  do
  {
    for ( j = 0; *(_DWORD *)v27 > j; ++j )
    {
      v11 = a2 + *v25 + 224LL * j;
      v12 = (pthread_mutex_t *)(v11 + 64);
      v32 = (_BYTE *)(v11 + 104);
      if ( pthread_mutex_trylock((pthread_mutex_t *)(v11 + 64)) )
      {
        sub_130AD90(v11);
        *(_BYTE *)(v11 + 104) = 1;
      }
      ++*(_QWORD *)(v11 + 56);
      if ( a1 != *(_QWORD *)(v11 + 48) )
      {
        ++*(_QWORD *)(v11 + 40);
        *(_QWORD *)(v11 + 48) = a1;
      }
      v13 = *(unsigned __int64 **)(v11 + 192);
      if ( v13 )
      {
        *(_QWORD *)(v11 + 192) = 0;
        *(_BYTE *)(v11 + 104) = 0;
        pthread_mutex_unlock(v12);
        sub_13152A0(a1, a2, v13);
        if ( pthread_mutex_trylock(v12) )
        {
          sub_130AD90(v11);
          *(_BYTE *)(v11 + 104) = 1;
        }
        ++*(_QWORD *)(v11 + 56);
        if ( a1 != *(_QWORD *)(v11 + 48) )
        {
          ++*(_QWORD *)(v11 + 40);
          *(_QWORD *)(v11 + 48) = a1;
        }
      }
      while ( 1 )
      {
        v14 = sub_133FAA0(v11 + 200);
        if ( !v14 )
          break;
        while ( 1 )
        {
          v31 = (unsigned __int64 *)v14;
          *v32 = 0;
          pthread_mutex_unlock(v12);
          sub_13152A0(a1, a2, v31);
          if ( pthread_mutex_trylock(v12) )
          {
            sub_130AD90(v11);
            *v32 = 1;
          }
          ++*(_QWORD *)(v11 + 56);
          if ( a1 == *(_QWORD *)(v11 + 48) )
            break;
          ++*(_QWORD *)(v11 + 40);
          *(_QWORD *)(v11 + 48) = a1;
          v14 = sub_133FAA0(v11 + 200);
          if ( !v14 )
            goto LABEL_33;
        }
      }
LABEL_33:
      while ( 1 )
      {
        v15 = *(unsigned __int64 **)(v11 + 216);
        if ( !v15 )
          break;
        while ( 1 )
        {
          if ( *(_DWORD *)(a2 + 78928) >= dword_5057900[0] )
          {
            v16 = (unsigned __int64 *)v15[5];
            if ( v15 == v16 )
            {
              *(_QWORD *)(v11 + 216) = 0;
            }
            else
            {
              *(_QWORD *)(v11 + 216) = v16;
              *(_QWORD *)(v15[6] + 40) = *(_QWORD *)(v15[5] + 48);
              v24 = v15[6];
              *(_QWORD *)(v15[5] + 48) = v24;
              v15[6] = *(_QWORD *)(v24 + 40);
              *(_QWORD *)(*(_QWORD *)(v15[5] + 48) + 40LL) = v15[5];
              *(_QWORD *)(v15[6] + 40) = v15;
            }
          }
          *v32 = 0;
          pthread_mutex_unlock(v12);
          sub_13152A0(a1, a2, v15);
          if ( pthread_mutex_trylock(v12) )
          {
            sub_130AD90(v11);
            *v32 = 1;
          }
          ++*(_QWORD *)(v11 + 56);
          if ( a1 == *(_QWORD *)(v11 + 48) )
            break;
          v15 = *(unsigned __int64 **)(v11 + 216);
          ++*(_QWORD *)(v11 + 40);
          *(_QWORD *)(v11 + 48) = a1;
          if ( !v15 )
            goto LABEL_41;
        }
      }
LABEL_41:
      *(_QWORD *)(v11 + 136) = 0;
      *(_QWORD *)(v11 + 176) = 0;
      *(_BYTE *)(v11 + 104) = 0;
      pthread_mutex_unlock(v12);
    }
    v27 += 40;
    ++v25;
  }
  while ( (char *)&unk_5260DE0 + 1460 != v27 );
  return sub_130B4A0(a1, a2 + 10648);
}
