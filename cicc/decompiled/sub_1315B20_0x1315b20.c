// Function: sub_1315B20
// Address: 0x1315b20
//
int __fastcall sub_1315B20(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v4; // rbx
  unsigned __int64 v5; // rcx
  unsigned __int64 *v6; // rax
  unsigned __int64 v7; // rsi
  _QWORD *v8; // rax
  unsigned __int64 *v9; // r12
  __int64 v10; // rbx
  int v11; // eax
  __int64 v12; // rdx
  _BYTE *v13; // r9
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  int result; // eax
  unsigned __int64 v18; // rdx
  _QWORD *v19; // rdx
  unsigned int i; // edi
  _QWORD *v21; // r10
  _QWORD *v22; // r8
  __int64 v23; // [rsp+0h] [rbp-1C0h]
  __int64 v24; // [rsp+8h] [rbp-1B8h]
  _BYTE *v25; // [rsp+8h] [rbp-1B8h]
  _BYTE *v26; // [rsp+8h] [rbp-1B8h]
  _QWORD v27[54]; // [rsp+10h] [rbp-1B0h] BYREF

  v4 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    v4 = v27;
    sub_130D500(v27);
  }
  v5 = a2 & 0xFFFFFFFFC0000000LL;
  v6 = (_QWORD *)((char *)v4 + ((a2 >> 26) & 0xF0));
  v7 = *v6;
  if ( (a2 & 0xFFFFFFFFC0000000LL) == *v6 )
  {
    v8 = (_QWORD *)(v6[1] + ((a2 >> 9) & 0x1FFFF8));
  }
  else if ( v5 == v4[32] )
  {
    v4[32] = v7;
    v18 = v4[33];
    v4[33] = v6[1];
LABEL_20:
    *v6 = v5;
    v6[1] = v18;
    v8 = (_QWORD *)(v18 + ((a2 >> 9) & 0x1FFFF8));
  }
  else
  {
    v19 = v4 + 34;
    for ( i = 1; i != 8; ++i )
    {
      if ( v5 == *v19 )
      {
        v21 = &v4[2 * i - 2];
        v22 = &v4[2 * i];
        v18 = v22[33];
        v22[32] = v21[32];
        v22[33] = v21[33];
        v21[32] = v7;
        v21[33] = v6[1];
        goto LABEL_20;
      }
      v19 += 2;
    }
    v8 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v4, a2, 1, 0);
  }
  v9 = (unsigned __int64 *)(((__int64)(*v8 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL);
  v23 = qword_50579C0[*v9 & 0xFFF];
  v24 = (unsigned __int8)(*v9 >> 20);
  v10 = dword_5060A40[v24] + v23 + 224 * ((*v9 >> 38) & 0x3F);
  v11 = pthread_mutex_trylock((pthread_mutex_t *)(v10 + 64));
  v12 = v24;
  v13 = (_BYTE *)(v10 + 104);
  if ( v11 )
  {
    sub_130AD90(v10);
    v13 = (_BYTE *)(v10 + 104);
    *(_BYTE *)(v10 + 104) = 1;
    v12 = v24;
  }
  ++*(_QWORD *)(v10 + 56);
  if ( a1 != *(_QWORD *)(v10 + 48) )
  {
    ++*(_QWORD *)(v10 + 40);
    *(_QWORD *)(v10 + 48) = a1;
  }
  v9[(((a2 - v9[1]) * dword_5260CA0[v12]) >> 38) + 8] ^= 1LL << (((a2 - v9[1]) * dword_5260CA0[v12]) >> 32);
  v14 = *v9 + 0x10000000;
  *v9 = v14;
  v15 = (v14 >> 28) & 0x3FF;
  if ( (_DWORD)v15 == *((_DWORD *)&unk_5260DE0 + 10 * v12 + 4) )
  {
    v25 = v13;
    sub_1315970(a1, v23, v9, (_QWORD *)v10);
    ++*(_QWORD *)(v10 + 120);
    --*(_QWORD *)(v10 + 136);
    *v25 = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(v10 + 64));
    result = sub_13152A0(a1, v23, v9);
  }
  else
  {
    if ( (_DWORD)v15 == 1 && *(unsigned __int64 **)(v10 + 192) != v9 )
    {
      v26 = v13;
      sub_1315A80(a1, v23, (__int64)v9, v10);
      v13 = v26;
    }
    ++*(_QWORD *)(v10 + 120);
    --*(_QWORD *)(v10 + 136);
    *v13 = 0;
    result = pthread_mutex_unlock((pthread_mutex_t *)(v10 + 64));
  }
  if ( a1 )
  {
    if ( --*(_DWORD *)(a1 + 152) < 0 )
    {
      result = sub_1314130((_DWORD *)(a1 + 152), (unsigned __int64 *)(a1 + 112));
      if ( (_BYTE)result )
        return sub_1315160(a1, v23, 0, 0);
    }
  }
  return result;
}
