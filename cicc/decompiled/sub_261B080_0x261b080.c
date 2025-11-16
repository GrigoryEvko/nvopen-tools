// Function: sub_261B080
// Address: 0x261b080
//
_QWORD *__fastcall sub_261B080(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  _BYTE *v6; // rsi
  _QWORD *v7; // r13
  _BYTE *v8; // rax
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  char *v11; // rcx
  size_t v12; // r15
  __int64 v13; // rax
  __m128i v14; // xmm1
  __int64 v15; // rdi
  int *v16; // r12
  _QWORD *i; // rbx
  _QWORD *v18; // r14
  _QWORD *v19; // rax
  unsigned __int64 v20; // r15
  char *v21; // rcx
  __int64 v22; // r15
  __m128i v23; // xmm0
  int v24; // eax
  __int64 v25; // rdi

  v4 = sub_22077B0(0x50u);
  v6 = *(_BYTE **)(a1 + 32);
  v7 = (_QWORD *)v4;
  v8 = *(_BYTE **)(a1 + 40);
  v7[4] = 0;
  v7[5] = 0;
  v7[6] = 0;
  v9 = v8 - v6;
  if ( v8 == v6 )
  {
    v12 = 0;
    v9 = 0;
    v11 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_20:
      sub_4261EA(80, v6, v5);
    v10 = sub_22077B0(v9);
    v6 = *(_BYTE **)(a1 + 32);
    v11 = (char *)v10;
    v8 = *(_BYTE **)(a1 + 40);
    v12 = v8 - v6;
  }
  v7[4] = v11;
  v7[5] = v11;
  v7[6] = &v11[v9];
  if ( v6 != v8 )
    v11 = (char *)memmove(v11, v6, v12);
  v13 = *(_QWORD *)(a1 + 72);
  v14 = _mm_loadu_si128((const __m128i *)(a1 + 56));
  v7[1] = a2;
  v15 = *(_QWORD *)(a1 + 24);
  v7[5] = &v11[v12];
  v7[9] = v13;
  LODWORD(v13) = *(_DWORD *)a1;
  v7[2] = 0;
  *(_DWORD *)v7 = v13;
  v7[3] = 0;
  *(__m128i *)(v7 + 7) = v14;
  if ( v15 )
  {
    v6 = v7;
    v7[3] = sub_261B080(v15, v7);
  }
  v16 = *(int **)(a1 + 16);
  for ( i = v7; v16; v16 = (int *)*((_QWORD *)v16 + 2) )
  {
    v18 = i;
    v19 = (_QWORD *)sub_22077B0(0x50u);
    v5 = *((_QWORD *)v16 + 5) - *((_QWORD *)v16 + 4);
    i = v19;
    v19[4] = 0;
    v20 = v5;
    v19[5] = 0;
    v19[6] = 0;
    if ( v5 )
    {
      if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_20;
      v21 = (char *)sub_22077B0(v5);
    }
    else
    {
      v21 = 0;
    }
    i[4] = v21;
    i[5] = v21;
    i[6] = &v21[v20];
    v6 = (_BYTE *)*((_QWORD *)v16 + 4);
    v22 = *((_QWORD *)v16 + 5) - (_QWORD)v6;
    if ( *((_BYTE **)v16 + 5) != v6 )
      v21 = (char *)memmove(v21, v6, *((_QWORD *)v16 + 5) - (_QWORD)v6);
    i[5] = &v21[v22];
    v23 = _mm_loadu_si128((const __m128i *)(v16 + 14));
    i[9] = *((_QWORD *)v16 + 9);
    *(__m128i *)(i + 7) = v23;
    v24 = *v16;
    i[2] = 0;
    *(_DWORD *)i = v24;
    i[3] = 0;
    v18[2] = i;
    i[1] = v18;
    v25 = *((_QWORD *)v16 + 3);
    if ( v25 )
    {
      v6 = i;
      i[3] = sub_261B080(v25, i);
    }
  }
  return v7;
}
