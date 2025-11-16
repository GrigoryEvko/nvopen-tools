// Function: sub_2CE0440
// Address: 0x2ce0440
//
_BYTE *__fastcall sub_2CE0440(_BYTE *a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  _BYTE *v10; // rax
  size_t v11; // r13
  char *v12; // rsi
  size_t v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rdi
  char *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  _BYTE *result; // rax
  _BYTE *v20; // rax
  unsigned __int64 v21; // r8
  char *v22; // rax
  char *v23; // rsi
  unsigned int v24; // ecx
  unsigned int v25; // eax
  __int64 v26; // rdi

  v6 = a3(a4, "MemorySpaceOptPass]", 18);
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
    v20 = *(_BYTE **)(a2 + 24);
    v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
    *(_QWORD *)(a2 + 32) = v8;
    if ( v20 != v8 )
      goto LABEL_4;
    goto LABEL_19;
  }
  if ( v10 != v8 )
  {
LABEL_4:
    *v8 = 60;
    ++*(_QWORD *)(a2 + 32);
    goto LABEL_5;
  }
LABEL_19:
  sub_CB6200(a2, "<", 1u);
LABEL_5:
  v12 = "second";
  if ( *a1 )
    v12 = "first";
  v13 = strlen(v12);
  v14 = *(_QWORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v14 < v13 )
  {
    v15 = sub_CB6200(a2, (unsigned __int8 *)v12, v13);
    v16 = *(char **)(v15 + 32);
    goto LABEL_9;
  }
  if ( (unsigned int)v13 >= 8 )
  {
    v21 = (unsigned __int64)(v14 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v14 = *(_QWORD *)v12;
    *(_QWORD *)((char *)v14 + (unsigned int)v13 - 8) = *(_QWORD *)&v12[(unsigned int)v13 - 8];
    v22 = (char *)v14 - v21;
    v23 = (char *)(v12 - v22);
    if ( (((_DWORD)v13 + (_DWORD)v22) & 0xFFFFFFF8) >= 8 )
    {
      v24 = (v13 + (_DWORD)v22) & 0xFFFFFFF8;
      v25 = 0;
      do
      {
        v26 = v25;
        v25 += 8;
        *(_QWORD *)(v21 + v26) = *(_QWORD *)&v23[v26];
      }
      while ( v25 < v24 );
    }
LABEL_28:
    v14 = *(_QWORD **)(a2 + 32);
    goto LABEL_29;
  }
  if ( (v13 & 4) != 0 )
  {
    *(_DWORD *)v14 = *(_DWORD *)v12;
    *(_DWORD *)((char *)v14 + (unsigned int)v13 - 4) = *(_DWORD *)&v12[(unsigned int)v13 - 4];
    v14 = *(_QWORD **)(a2 + 32);
    goto LABEL_29;
  }
  if ( (_DWORD)v13 )
  {
    *(_BYTE *)v14 = *v12;
    if ( (v13 & 2) != 0 )
    {
      *(_WORD *)((char *)v14 + (unsigned int)v13 - 2) = *(_WORD *)&v12[(unsigned int)v13 - 2];
      v14 = *(_QWORD **)(a2 + 32);
      goto LABEL_29;
    }
    goto LABEL_28;
  }
LABEL_29:
  v16 = (char *)v14 + v13;
  v15 = a2;
  *(_QWORD *)(a2 + 32) = v16;
LABEL_9:
  if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 5u )
  {
    sub_CB6200(v15, "-time;", 6u);
  }
  else
  {
    *(_DWORD *)v16 = 1835627565;
    *((_WORD *)v16 + 2) = 15205;
    *(_QWORD *)(v15 + 32) += 6LL;
  }
  v17 = *(_QWORD *)(a2 + 32);
  v18 = a2;
  if ( !a1[1] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v17) > 2 )
    {
      *(_BYTE *)(v17 + 2) = 45;
      *(_WORD *)v17 = 28526;
      v17 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v17;
    }
    else
    {
      v18 = sub_CB6200(a2, "no-", 3u);
      v17 = *(_QWORD *)(v18 + 32);
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(v18 + 24) - v17) <= 7 )
  {
    sub_CB6200(v18, (unsigned __int8 *)"warnings", 8u);
  }
  else
  {
    *(_QWORD *)v17 = 0x73676E696E726177LL;
    *(_QWORD *)(v18 + 32) += 8LL;
  }
  result = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == result )
    return (_BYTE *)sub_CB6200(a2, (unsigned __int8 *)">", 1u);
  *result = 62;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
