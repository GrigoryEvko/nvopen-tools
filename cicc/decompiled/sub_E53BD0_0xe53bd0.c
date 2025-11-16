// Function: sub_E53BD0
// Address: 0xe53bd0
//
_BYTE *__fastcall sub_E53BD0(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        unsigned int **a7,
        __int64 a8)
{
  __int64 v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rax
  unsigned int *v19; // r14
  __int64 v20; // rbx
  __int64 v21; // rdi
  _BYTE *v22; // rax
  unsigned __int64 v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // r13
  _BYTE *v27; // rax
  size_t *v28; // rsi
  void *v29; // rdi
  size_t v30; // r14
  void *v31; // rsi
  __int64 v33; // rax

  v12 = sub_904010(*(_QWORD *)(a1 + 304), "\t.pseudoprobe\t");
  v13 = sub_CB59D0(v12, a2);
  v14 = *(_BYTE **)(v13 + 32);
  if ( *(_BYTE **)(v13 + 24) == v14 )
  {
    v13 = sub_CB6200(v13, (unsigned __int8 *)" ", 1u);
  }
  else
  {
    *v14 = 32;
    ++*(_QWORD *)(v13 + 32);
  }
  v15 = sub_CB59D0(v13, a3);
  v16 = sub_904010(v15, " ");
  v17 = sub_CB59D0(v16, a4);
  v18 = *(_BYTE **)(v17 + 32);
  if ( *(_BYTE **)(v17 + 24) == v18 )
  {
    v17 = sub_CB6200(v17, (unsigned __int8 *)" ", 1u);
  }
  else
  {
    *v18 = 32;
    ++*(_QWORD *)(v17 + 32);
  }
  sub_CB59D0(v17, a5);
  if ( a6 )
  {
    v33 = sub_904010(*(_QWORD *)(a1 + 304), " ");
    sub_CB59D0(v33, a6);
  }
  v19 = *a7;
  v20 = (__int64)&(*a7)[4 * *((unsigned int *)a7 + 2)];
  if ( (unsigned int *)v20 != *a7 )
  {
    do
    {
      v24 = *(_QWORD *)(a1 + 304);
      v25 = *(_QWORD *)(v24 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v24 + 24) - v25) > 2 )
      {
        *(_BYTE *)(v25 + 2) = 32;
        *(_WORD *)v25 = 16416;
        *(_QWORD *)(v24 + 32) += 3LL;
      }
      else
      {
        v24 = sub_CB6200(v24, " @ ", 3u);
      }
      v21 = sub_CB59D0(v24, *((_QWORD *)v19 + 1));
      v22 = *(_BYTE **)(v21 + 32);
      if ( *(_BYTE **)(v21 + 24) == v22 )
      {
        v21 = sub_CB6200(v21, (unsigned __int8 *)":", 1u);
      }
      else
      {
        *v22 = 58;
        ++*(_QWORD *)(v21 + 32);
      }
      v23 = *v19;
      v19 += 4;
      sub_CB59D0(v21, v23);
    }
    while ( (unsigned int *)v20 != v19 );
  }
  v26 = *(_QWORD *)(a1 + 304);
  v27 = *(_BYTE **)(v26 + 32);
  if ( *(_BYTE **)(v26 + 24) == v27 )
  {
    v26 = sub_CB6200(*(_QWORD *)(a1 + 304), (unsigned __int8 *)" ", 1u);
  }
  else
  {
    *v27 = 32;
    ++*(_QWORD *)(v26 + 32);
  }
  if ( (*(_BYTE *)(a8 + 8) & 1) != 0 )
  {
    v28 = *(size_t **)(a8 - 8);
    v29 = *(void **)(v26 + 32);
    v30 = *v28;
    v31 = v28 + 3;
    if ( *(_QWORD *)(v26 + 24) - (_QWORD)v29 >= v30 )
    {
      if ( v30 )
      {
        memcpy(v29, v31, v30);
        *(_QWORD *)(v26 + 32) += v30;
      }
    }
    else
    {
      sub_CB6200(v26, (unsigned __int8 *)v31, v30);
    }
  }
  return sub_E4D880(a1);
}
