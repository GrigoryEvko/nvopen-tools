// Function: sub_1339F60
// Address: 0x1339f60
//
__int64 __fastcall sub_1339F60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _DWORD *a4,
        __int64 *a5,
        unsigned __int64 *a6,
        __int64 a7)
{
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // r15
  __int64 v13; // r13
  _QWORD *v14; // r10
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rsi
  _QWORD *v17; // rcx
  __int64 v18; // rax
  int v19; // eax
  __int64 v20; // rdx
  unsigned int v21; // r12d
  unsigned int v22; // eax
  __int64 v23; // rcx
  unsigned __int64 v25; // rdx
  _QWORD *v26; // rdx
  unsigned int i; // edi
  _QWORD *v28; // r8
  unsigned __int64 v29; // [rsp+0h] [rbp-1C0h]
  _QWORD v30[54]; // [rsp+10h] [rbp-1B0h] BYREF

  if ( pthread_mutex_trylock(&stru_4F96C00) )
  {
    sub_130AD90((__int64)&xmmword_4F96BC0);
    byte_4F96C28 = 1;
  }
  ++*((_QWORD *)&xmmword_4F96BF0 + 1);
  if ( a1 != (_QWORD)xmmword_4F96BF0 )
  {
    ++qword_4F96BE8;
    *(_QWORD *)&xmmword_4F96BF0 = a1;
  }
  if ( a6 )
  {
    if ( a7 != 8 )
    {
LABEL_24:
      v21 = 22;
      goto LABEL_25;
    }
    v11 = *a6;
    v12 = *a6 & 0xFFFFFFFFC0000000LL;
    v13 = (*a6 >> 30) & 0xF;
  }
  else
  {
    v12 = 0;
    v13 = 0;
    v11 = 0;
  }
  v14 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    v29 = v11;
    sub_130D500(v30);
    v14 = v30;
    v11 = v29;
  }
  v15 = &v14[2 * v13];
  v16 = *v15;
  if ( *v15 == v12 )
  {
    v17 = (_QWORD *)(v15[1] + ((v11 >> 9) & 0x1FFFF8));
  }
  else if ( v14[32] == v12 )
  {
    v25 = v14[33];
LABEL_30:
    v14[32] = v16;
    v14[33] = v15[1];
    v17 = (_QWORD *)(v25 + ((v11 >> 9) & 0x1FFFF8));
    *v15 = v12;
    v15[1] = v25;
  }
  else
  {
    v26 = v14 + 34;
    for ( i = 1; i != 8; ++i )
    {
      if ( *v26 == v12 )
      {
        v28 = &v14[2 * i];
        v14 += 2 * i - 2;
        v25 = v28[33];
        v28[32] = v14[32];
        v28[33] = v14[33];
        goto LABEL_30;
      }
      v26 += 2;
    }
    v17 = (_QWORD *)sub_130D370(a1, (__int64)&unk_5060AE0, v14, v11, 1, 0);
  }
  if ( (((__int64)(*v17 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL) == 0 )
    goto LABEL_24;
  v18 = qword_50579C0[*(_QWORD *)(((__int64)(*v17 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL) & 0xFFFLL];
  if ( !v18 )
    goto LABEL_24;
  v19 = *(_DWORD *)(v18 + 78928);
  LODWORD(v30[0]) = v19;
  if ( !a4 || !a5 )
  {
    v21 = 0;
    goto LABEL_25;
  }
  v20 = *a5;
  if ( *a5 != 4 )
  {
    if ( (unsigned __int64)*a5 > 4 )
      v20 = 4;
    if ( (_DWORD)v20 )
    {
      v22 = 0;
      do
      {
        v23 = v22++;
        *((_BYTE *)a4 + v23) = *((_BYTE *)v30 + v23);
      }
      while ( v22 < (unsigned int)v20 );
    }
    *a5 = v20;
    goto LABEL_24;
  }
  *a4 = v19;
  v21 = 0;
LABEL_25:
  byte_4F96C28 = 0;
  pthread_mutex_unlock(&stru_4F96C00);
  return v21;
}
