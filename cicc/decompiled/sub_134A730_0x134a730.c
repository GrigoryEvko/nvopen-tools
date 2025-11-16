// Function: sub_134A730
// Address: 0x134a730
//
int __fastcall sub_134A730(
        __int64 a1,
        unsigned __int64 a2,
        __int64 *a3,
        _QWORD *a4,
        unsigned __int64 *a5,
        _QWORD *a6,
        _QWORD *a7,
        _QWORD *a8)
{
  _QWORD *v9; // rdx
  _QWORD *v14; // r11
  _QWORD *v15; // r10
  unsigned __int64 v16; // rcx
  unsigned __int64 *v17; // rax
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rsi
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r13
  int v23; // eax
  _QWORD *v24; // r11
  _QWORD *v25; // r10
  _BYTE *v26; // rdx
  __int64 v27; // rax
  _QWORD *v28; // rax
  unsigned __int64 v29; // r8
  _QWORD *v30; // r9
  unsigned int i; // r8d
  _QWORD *v32; // r9
  _QWORD *v33; // rdx
  unsigned __int64 v34; // rax
  _QWORD *v36; // [rsp+10h] [rbp-1C0h]
  _QWORD *v37; // [rsp+10h] [rbp-1C0h]
  _QWORD *v38; // [rsp+10h] [rbp-1C0h]
  _QWORD *v39; // [rsp+18h] [rbp-1B8h]
  _BYTE *v40; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v41; // [rsp+18h] [rbp-1B8h]
  _QWORD *v42; // [rsp+18h] [rbp-1B8h]
  _QWORD v43[54]; // [rsp+20h] [rbp-1B0h] BYREF

  v9 = (_QWORD *)(a1 + 432);
  v14 = a7;
  v15 = a8;
  if ( !a1 )
  {
    sub_130D500(v43);
    v9 = v43;
    v15 = a8;
    v14 = a7;
  }
  v16 = a2 & 0xFFFFFFFFC0000000LL;
  v17 = (_QWORD *)((char *)v9 + ((a2 >> 26) & 0xF0));
  v18 = *v17;
  if ( (a2 & 0xFFFFFFFFC0000000LL) == *v17 )
  {
    v19 = (_QWORD *)(v17[1] + ((a2 >> 9) & 0x1FFFF8));
  }
  else if ( v16 == v9[32] )
  {
    v29 = v9[33];
    v9[32] = v18;
    v19 = (_QWORD *)(v29 + ((a2 >> 9) & 0x1FFFF8));
    v9[33] = v17[1];
    *v17 = v16;
    v17[1] = v29;
  }
  else
  {
    v30 = v9 + 34;
    for ( i = 1; i != 8; ++i )
    {
      if ( v16 == *v30 )
      {
        v32 = &v9[2 * i];
        v41 = v32[33];
        v33 = &v9[2 * i - 2];
        v32[32] = v33[32];
        v32[33] = v33[33];
        v33[32] = v18;
        v33[33] = v17[1];
        *v17 = v16;
        v17[1] = v41;
        v19 = (_QWORD *)(v41 + ((a2 >> 9) & 0x1FFFF8));
        goto LABEL_5;
      }
      v30 += 2;
    }
    v38 = v14;
    v42 = v15;
    v34 = sub_130D370(a1, (__int64)&unk_5060AE0, v9, a2, 1, 0);
    v14 = v38;
    v15 = v42;
    v19 = (_QWORD *)v34;
  }
LABEL_5:
  v20 = (_QWORD *)(((__int64)(*v19 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL);
  if ( v20 )
  {
    *a5 = *(_QWORD *)((((__int64)(*v19 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL) + 0x10) & 0xFFFFFFFFFFFFF000LL;
    if ( (*v20 & 0x1000LL) != 0 )
    {
      v36 = v15;
      v39 = v14;
      *a3 = (*v20 >> 28) & 0x3FFLL;
      v21 = (unsigned __int8)(*v20 >> 20);
      *a4 = *((unsigned int *)&unk_5260DE0 + 10 * v21 + 4);
      v22 = dword_5060A40[v21] + qword_50579C0[*v20 & 0xFFFLL] + 224 * ((*v20 >> 38) & 0x3FLL);
      v23 = pthread_mutex_trylock((pthread_mutex_t *)(v22 + 64));
      v24 = v39;
      v25 = v36;
      v26 = (_BYTE *)(v22 + 104);
      if ( v23 )
      {
        sub_130AD90(v22);
        v26 = (_BYTE *)(v22 + 104);
        *(_BYTE *)(v22 + 104) = 1;
        v25 = v36;
        v24 = v39;
      }
      ++*(_QWORD *)(v22 + 56);
      if ( a1 != *(_QWORD *)(v22 + 48) )
      {
        ++*(_QWORD *)(v22 + 40);
        *(_QWORD *)(v22 + 48) = a1;
      }
      v27 = *(_QWORD *)(v22 + 176) * *a4;
      *v24 = v27;
      *a6 = v27 - *(_QWORD *)(v22 + 136);
      v28 = *(_QWORD **)(v22 + 192);
      if ( v28 || (v37 = v25, v40 = v26, v28 = sub_133F530((_QWORD *)(v22 + 200)), v26 = v40, v25 = v37, v28) )
        v28 = (_QWORD *)v28[1];
      *v25 = v28;
      *v26 = 0;
      LODWORD(v20) = pthread_mutex_unlock((pthread_mutex_t *)(v22 + 64));
    }
    else
    {
      *v14 = 0;
      *a6 = 0;
      *a3 = 0;
      *a4 = 1;
      *v15 = 0;
    }
  }
  else
  {
    *v14 = 0;
    *a6 = 0;
    *a5 = 0;
    *a4 = 0;
    *a3 = 0;
    *v15 = 0;
  }
  return (int)v20;
}
