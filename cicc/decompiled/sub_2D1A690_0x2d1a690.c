// Function: sub_2D1A690
// Address: 0x2d1a690
//
_BYTE *__fastcall sub_2D1A690(_BYTE *a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  _BYTE *v10; // rax
  size_t v11; // r13
  char *v12; // r13
  size_t v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rdi
  char *v16; // rax
  char *v17; // r13
  size_t v18; // rdx
  _QWORD *v19; // rax
  __int64 v20; // rdi
  char *v21; // rax
  _BYTE *result; // rax
  _BYTE *v23; // rax
  unsigned __int64 v24; // rdi
  char *v25; // rax
  char *v26; // r13
  unsigned int v27; // ecx
  unsigned int v28; // eax
  __int64 v29; // rsi
  unsigned __int64 v30; // r8
  char *v31; // rax
  char *v32; // rsi
  unsigned int v33; // ecx
  unsigned int v34; // eax
  __int64 v35; // rdi

  v6 = a3(a4, "SetGlobalArrayAlignmentPass]", 27);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_BYTE **)(a2 + 24);
  v11 = v7;
  if ( v10 - v8 < v7 )
  {
    sub_CB6200(a2, v9, v7);
    v10 = *(_BYTE **)(a2 + 24);
    v8 = *(_BYTE **)(a2 + 32);
  }
  else if ( v7 )
  {
    memcpy(v8, v9, v7);
    v23 = *(_BYTE **)(a2 + 24);
    v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
    *(_QWORD *)(a2 + 32) = v8;
    if ( v8 != v23 )
      goto LABEL_4;
    goto LABEL_20;
  }
  if ( v8 != v10 )
  {
LABEL_4:
    *v8 = 60;
    ++*(_QWORD *)(a2 + 32);
    goto LABEL_5;
  }
LABEL_20:
  sub_CB6200(a2, "<", 1u);
LABEL_5:
  v12 = "modify";
  if ( !*a1 )
    v12 = "skip";
  v13 = strlen(v12);
  v14 = *(_QWORD **)(a2 + 32);
  if ( v13 > *(_QWORD *)(a2 + 24) - (_QWORD)v14 )
  {
    v15 = sub_CB6200(a2, (unsigned __int8 *)v12, v13);
    v16 = *(char **)(v15 + 32);
    goto LABEL_9;
  }
  if ( (unsigned int)v13 >= 8 )
  {
    v30 = (unsigned __int64)(v14 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v14 = *(_QWORD *)v12;
    *(_QWORD *)((char *)v14 + (unsigned int)v13 - 8) = *(_QWORD *)&v12[(unsigned int)v13 - 8];
    v31 = (char *)v14 - v30;
    v32 = (char *)(v12 - v31);
    if ( (((_DWORD)v13 + (_DWORD)v31) & 0xFFFFFFF8) >= 8 )
    {
      v33 = (v13 + (_DWORD)v31) & 0xFFFFFFF8;
      v34 = 0;
      do
      {
        v35 = v34;
        v34 += 8;
        *(_QWORD *)(v30 + v35) = *(_QWORD *)&v32[v35];
      }
      while ( v34 < v33 );
    }
LABEL_35:
    v14 = *(_QWORD **)(a2 + 32);
    goto LABEL_36;
  }
  if ( (v13 & 4) != 0 )
  {
    *(_DWORD *)v14 = *(_DWORD *)v12;
    *(_DWORD *)((char *)v14 + (unsigned int)v13 - 4) = *(_DWORD *)&v12[(unsigned int)v13 - 4];
    v14 = *(_QWORD **)(a2 + 32);
    goto LABEL_36;
  }
  if ( (_DWORD)v13 )
  {
    *(_BYTE *)v14 = *v12;
    if ( (v13 & 2) != 0 )
    {
      *(_WORD *)((char *)v14 + (unsigned int)v13 - 2) = *(_WORD *)&v12[(unsigned int)v13 - 2];
      v14 = *(_QWORD **)(a2 + 32);
      goto LABEL_36;
    }
    goto LABEL_35;
  }
LABEL_36:
  v16 = (char *)v14 + v13;
  v15 = a2;
  *(_QWORD *)(a2 + 32) = v16;
LABEL_9:
  if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 0xBu )
  {
    sub_CB6200(v15, "-shared-mem;", 0xCu);
  }
  else
  {
    qmemcpy(v16, "-shared-mem;", 12);
    *(_QWORD *)(v15 + 32) += 12LL;
  }
  v17 = "modify";
  if ( !a1[1] )
    v17 = "skip";
  v18 = strlen(v17);
  v19 = *(_QWORD **)(a2 + 32);
  if ( v18 > *(_QWORD *)(a2 + 24) - (_QWORD)v19 )
  {
    v20 = sub_CB6200(a2, (unsigned __int8 *)v17, v18);
    v21 = *(char **)(v20 + 32);
    goto LABEL_15;
  }
  if ( (unsigned int)v18 >= 8 )
  {
    v24 = (unsigned __int64)(v19 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v19 = *(_QWORD *)v17;
    *(_QWORD *)((char *)v19 + (unsigned int)v18 - 8) = *(_QWORD *)&v17[(unsigned int)v18 - 8];
    v25 = (char *)v19 - v24;
    v26 = (char *)(v17 - v25);
    if ( (((_DWORD)v18 + (_DWORD)v25) & 0xFFFFFFF8) >= 8 )
    {
      v27 = (v18 + (_DWORD)v25) & 0xFFFFFFF8;
      v28 = 0;
      do
      {
        v29 = v28;
        v28 += 8;
        *(_QWORD *)(v24 + v29) = *(_QWORD *)&v26[v29];
      }
      while ( v28 < v27 );
    }
LABEL_29:
    v19 = *(_QWORD **)(a2 + 32);
    goto LABEL_30;
  }
  if ( (v18 & 4) != 0 )
  {
    *(_DWORD *)v19 = *(_DWORD *)v17;
    *(_DWORD *)((char *)v19 + (unsigned int)v18 - 4) = *(_DWORD *)&v17[(unsigned int)v18 - 4];
    v19 = *(_QWORD **)(a2 + 32);
    goto LABEL_30;
  }
  if ( (_DWORD)v18 )
  {
    *(_BYTE *)v19 = *v17;
    if ( (v18 & 2) != 0 )
    {
      *(_WORD *)((char *)v19 + (unsigned int)v18 - 2) = *(_WORD *)&v17[(unsigned int)v18 - 2];
      v19 = *(_QWORD **)(a2 + 32);
      goto LABEL_30;
    }
    goto LABEL_29;
  }
LABEL_30:
  v21 = (char *)v19 + v18;
  v20 = a2;
  *(_QWORD *)(a2 + 32) = v21;
LABEL_15:
  if ( *(_QWORD *)(v20 + 24) - (_QWORD)v21 <= 0xAu )
  {
    sub_CB6200(v20, "-global-mem", 0xBu);
  }
  else
  {
    qmemcpy(v21, "-global-mem", 11);
    *(_QWORD *)(v20 + 32) += 11LL;
  }
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)">", 1u);
  *result = 62;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
