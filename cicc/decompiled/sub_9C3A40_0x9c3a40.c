// Function: sub_9C3A40
// Address: 0x9c3a40
//
_QWORD *__fastcall sub_9C3A40(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // rcx
  _BYTE *v7; // rsi
  _QWORD *v8; // r13
  _BYTE *v9; // rax
  unsigned __int64 v10; // r14
  __int64 v11; // rax
  char *v12; // rcx
  size_t v13; // r15
  __int64 v14; // rax
  __m128i v15; // xmm1
  __int64 v16; // rdi
  int *v17; // r12
  _QWORD *i; // rbx
  _QWORD *v19; // r14
  _QWORD *v20; // rax
  unsigned __int64 v21; // r15
  char *v22; // rcx
  __int64 v23; // r15
  __m128i v24; // xmm0
  int v25; // eax
  __int64 v26; // rdi

  v4 = sub_22077B0(80);
  v7 = *(_BYTE **)(a1 + 32);
  v8 = (_QWORD *)v4;
  v9 = *(_BYTE **)(a1 + 40);
  v8[4] = 0;
  v8[5] = 0;
  v8[6] = 0;
  v10 = v9 - v7;
  if ( v9 == v7 )
  {
    v13 = 0;
    v10 = 0;
    v12 = 0;
  }
  else
  {
    if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_20:
      sub_4261EA(80, v7, v5, v6);
    v11 = sub_22077B0(v10);
    v7 = *(_BYTE **)(a1 + 32);
    v12 = (char *)v11;
    v9 = *(_BYTE **)(a1 + 40);
    v13 = v9 - v7;
  }
  v8[4] = v12;
  v8[5] = v12;
  v8[6] = &v12[v10];
  if ( v7 != v9 )
    v12 = (char *)memmove(v12, v7, v13);
  v14 = *(_QWORD *)(a1 + 72);
  v15 = _mm_loadu_si128((const __m128i *)(a1 + 56));
  v8[1] = a2;
  v16 = *(_QWORD *)(a1 + 24);
  v8[5] = &v12[v13];
  v8[9] = v14;
  LODWORD(v14) = *(_DWORD *)a1;
  v8[2] = 0;
  *(_DWORD *)v8 = v14;
  v8[3] = 0;
  *(__m128i *)(v8 + 7) = v15;
  if ( v16 )
  {
    v7 = v8;
    v8[3] = sub_9C3A40(v16, v8);
  }
  v17 = *(int **)(a1 + 16);
  for ( i = v8; v17; v17 = (int *)*((_QWORD *)v17 + 2) )
  {
    v19 = i;
    v20 = (_QWORD *)sub_22077B0(80);
    v5 = *((_QWORD *)v17 + 5) - *((_QWORD *)v17 + 4);
    i = v20;
    v20[4] = 0;
    v21 = v5;
    v20[5] = 0;
    v20[6] = 0;
    if ( v5 )
    {
      if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_20;
      v22 = (char *)sub_22077B0(v5);
    }
    else
    {
      v22 = 0;
    }
    i[4] = v22;
    i[5] = v22;
    i[6] = &v22[v21];
    v7 = (_BYTE *)*((_QWORD *)v17 + 4);
    v23 = *((_QWORD *)v17 + 5) - (_QWORD)v7;
    if ( *((_BYTE **)v17 + 5) != v7 )
      v22 = (char *)memmove(v22, v7, *((_QWORD *)v17 + 5) - (_QWORD)v7);
    i[5] = &v22[v23];
    v24 = _mm_loadu_si128((const __m128i *)(v17 + 14));
    i[9] = *((_QWORD *)v17 + 9);
    *(__m128i *)(i + 7) = v24;
    v25 = *v17;
    i[2] = 0;
    *(_DWORD *)i = v25;
    i[3] = 0;
    v19[2] = i;
    i[1] = v19;
    v26 = *((_QWORD *)v17 + 3);
    if ( v26 )
    {
      v7 = i;
      i[3] = sub_9C3A40(v26, i);
    }
  }
  return v8;
}
