// Function: sub_1316B20
// Address: 0x1316b20
//
unsigned __int64 __fastcall sub_1316B20(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int64 a5,
        char a6)
{
  __int64 v6; // r15
  __int64 v7; // r14
  unsigned __int64 v8; // r12
  __int64 *v9; // r13
  __int64 v10; // rbx
  __int64 *v11; // rbx
  _QWORD *i; // rax
  _QWORD *v13; // r14
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // r13
  unsigned __int64 v16; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // [rsp+0h] [rbp-A0h]
  __int64 v22; // [rsp+8h] [rbp-98h]
  unsigned int v23; // [rsp+10h] [rbp-90h]
  unsigned int v24; // [rsp+14h] [rbp-8Ch]
  __int64 v25; // [rsp+18h] [rbp-88h]
  __int64 v29; // [rsp+30h] [rbp-70h]
  void **v30; // [rsp+48h] [rbp-58h]
  unsigned __int64 v31; // [rsp+50h] [rbp-50h]
  unsigned int v33[13]; // [rsp+6Ch] [rbp-34h] BYREF

  v6 = a2;
  v7 = a1;
  v8 = 0;
  v9 = (__int64 *)((char *)&unk_5260DE0 + 40 * a3);
  v24 = *(_DWORD *)(a2 + 78928);
  v10 = 0;
  v31 = *((unsigned int *)v9 + 4);
  v22 = *v9;
  v23 = dword_5057900[0];
  v21 = 0;
  v25 = sub_1315920(a1, a2, a3, v33);
  v29 = 0;
  if ( a5 )
  {
    v11 = v9;
    for ( i = sub_1316650(a1, a2, a3, v33[0], (__int64)v9); ; i = sub_1316650(a1, a2, a3, v33[0], (__int64)v11) )
    {
      v13 = i;
      if ( !i )
        break;
      ++v29;
      v14 = a5 - v8;
      v15 = v31;
      if ( v31 > a5 - v8 )
        v15 = a5 - v8;
      v30 = (void **)(a4 + 8 * v8);
      sub_1314190(i, v11, v15, (__int64)v30);
      if ( a6 )
        memset(*v30, 0, v15 * v22);
      v8 += v15;
      if ( v31 > v14 )
      {
        if ( a5 <= v8 )
          break;
      }
      else
      {
        if ( v24 >= v23 )
        {
          v13[5] = v13;
          v13[6] = v13;
          if ( v21 )
          {
            v13[5] = v21[6];
            v21[6] = v13;
            v13[6] = *(_QWORD *)(v13[6] + 40LL);
            *(_QWORD *)(v21[6] + 40LL) = v21;
            *(_QWORD *)(v13[6] + 40LL) = v13;
            v21 = (_QWORD *)v13[5];
          }
          else
          {
            v21 = v13;
          }
        }
        v13 = 0;
        if ( a5 <= v8 )
          break;
      }
    }
    v10 = (__int64)v13;
    v6 = a2;
    v7 = a1;
  }
  if ( pthread_mutex_trylock((pthread_mutex_t *)(v25 + 64)) )
  {
    sub_130AD90(v25);
    *(_BYTE *)(v25 + 104) = 1;
  }
  ++*(_QWORD *)(v25 + 56);
  if ( v7 != *(_QWORD *)(v25 + 48) )
  {
    ++*(_QWORD *)(v25 + 40);
    *(_QWORD *)(v25 + 48) = v7;
  }
  if ( v10 )
    sub_13142A0(v6, v10, (_QWORD *)v25, v16);
  if ( v24 >= v23 )
  {
    v19 = *(_QWORD *)(v25 + 216);
    if ( v19 )
    {
      if ( v21 )
      {
        *(_QWORD *)(v21[6] + 40LL) = *(_QWORD *)(v19 + 48);
        v20 = v21[6];
        *(_QWORD *)(*(_QWORD *)(v25 + 216) + 48LL) = v20;
        v21[6] = *(_QWORD *)(v20 + 40);
        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v25 + 216) + 48LL) + 40LL) = *(_QWORD *)(v25 + 216);
        *(_QWORD *)(v21[6] + 40LL) = v21;
      }
    }
    else
    {
      *(_QWORD *)(v25 + 216) = v21;
    }
  }
  *(_QWORD *)(v25 + 160) += v29;
  *(_QWORD *)(v25 + 176) += v29;
  *(_QWORD *)(v25 + 112) += v8;
  *(_QWORD *)(v25 + 128) += v8;
  *(_QWORD *)(v25 + 136) += v8;
  *(_BYTE *)(v25 + 104) = 0;
  pthread_mutex_unlock((pthread_mutex_t *)(v25 + 64));
  if ( v7 )
  {
    if ( --*(_DWORD *)(v7 + 152) < 0 )
    {
      if ( (unsigned __int8)sub_1314130((_DWORD *)(v7 + 152), (unsigned __int64 *)(v7 + 112)) )
        sub_1315160(v7, v6, 0, 0);
    }
  }
  return v8;
}
