// Function: sub_1635030
// Address: 0x1635030
//
__int64 __fastcall sub_1635030(__int64 a1, const char *a2, size_t a3, const char *a4, size_t a5)
{
  const char *v5; // r9
  unsigned int v6; // r15d
  int v9; // ebx
  const char *v10; // rdi
  size_t v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  const char *v16; // r9
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  __int64 v22; // rcx
  _BYTE *v23; // rax
  __int64 v24; // rax
  _WORD *v25; // rdx
  __int64 v26; // rbx
  _DWORD *v27; // rdi
  unsigned __int64 v28; // rax
  _BYTE *v29; // rdi
  _BYTE *v30; // rax
  _BYTE *v32; // rax
  unsigned __int64 v33; // rax
  unsigned int v34; // edx
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  const char *v40; // [rsp+18h] [rbp-38h]

  v5 = byte_3F871B3;
  v6 = 1;
  v9 = *(_DWORD *)(a1 + 12) + 1;
  *(_DWORD *)(a1 + 12) = v9;
  if ( dword_4F9EE20 != -1 )
  {
    if ( v9 > dword_4F9EE20 )
      v5 = "NOT ";
    LOBYTE(v6) = v9 <= dword_4F9EE20;
  }
  v10 = v5;
  v40 = v5;
  v11 = strlen(v5);
  v13 = sub_16E8CB0(v10, a2, v12);
  v16 = v40;
  v17 = *(_QWORD **)(v13 + 24);
  v18 = v13;
  if ( *(_QWORD *)(v13 + 16) - (_QWORD)v17 <= 7u )
  {
    v36 = sub_16E7EE0(v13, "BISECT: ", 8, v14, v15, v40);
    v16 = v40;
    v18 = v36;
    v19 = *(_BYTE **)(v36 + 24);
  }
  else
  {
    *v17 = 0x203A544345534942LL;
    v19 = (_BYTE *)(*(_QWORD *)(v13 + 24) + 8LL);
    *(_QWORD *)(v18 + 24) = v19;
  }
  v20 = *(_QWORD *)(v18 + 16);
  v21 = v20 - (_QWORD)v19;
  if ( v11 > v20 - (__int64)v19 )
  {
    v18 = sub_16E7EE0(v18, v16, v11);
    v19 = *(_BYTE **)(v18 + 24);
    v21 = *(_QWORD *)(v18 + 16) - (_QWORD)v19;
LABEL_9:
    if ( v21 > 0xC )
      goto LABEL_10;
LABEL_32:
    v18 = sub_16E7EE0(v18, "running pass ", 13);
    v23 = *(_BYTE **)(v18 + 24);
    if ( *(_BYTE **)(v18 + 16) != v23 )
      goto LABEL_11;
    goto LABEL_33;
  }
  if ( !v11 )
    goto LABEL_9;
  if ( (_DWORD)v11 )
  {
    v34 = 0;
    do
    {
      v35 = v34++;
      v19[v35] = v16[v35];
    }
    while ( v34 < (unsigned int)v11 );
    v20 = *(_QWORD *)(v18 + 16);
  }
  v19 = (_BYTE *)(v11 + *(_QWORD *)(v18 + 24));
  *(_QWORD *)(v18 + 24) = v19;
  if ( (unsigned __int64)(v20 - (_QWORD)v19) <= 0xC )
    goto LABEL_32;
LABEL_10:
  v22 = 0x20676E696E6E7572LL;
  qmemcpy(v19, "running pass ", 13);
  v23 = (_BYTE *)(*(_QWORD *)(v18 + 24) + 13LL);
  *(_QWORD *)(v18 + 24) = v23;
  if ( *(_BYTE **)(v18 + 16) != v23 )
  {
LABEL_11:
    *v23 = 40;
    ++*(_QWORD *)(v18 + 24);
    goto LABEL_12;
  }
LABEL_33:
  v18 = sub_16E7EE0(v18, "(", 1, v22, v15, v16);
LABEL_12:
  v24 = sub_16E7AB0(v18, v9);
  v25 = *(_WORD **)(v24 + 24);
  v26 = v24;
  if ( *(_QWORD *)(v24 + 16) - (_QWORD)v25 <= 1u )
  {
    v38 = sub_16E7EE0(v24, ") ", 2);
    v27 = *(_DWORD **)(v38 + 24);
    v26 = v38;
  }
  else
  {
    *v25 = 8233;
    v27 = (_DWORD *)(*(_QWORD *)(v24 + 24) + 2LL);
    *(_QWORD *)(v24 + 24) = v27;
  }
  v28 = *(_QWORD *)(v26 + 16) - (_QWORD)v27;
  if ( v28 < a3 )
  {
    v37 = sub_16E7EE0(v26, a2, a3);
    v27 = *(_DWORD **)(v37 + 24);
    v26 = v37;
    v28 = *(_QWORD *)(v37 + 16) - (_QWORD)v27;
LABEL_16:
    if ( v28 > 3 )
      goto LABEL_17;
    goto LABEL_25;
  }
  if ( !a3 )
    goto LABEL_16;
  memcpy(v27, a2, a3);
  v27 = (_DWORD *)(a3 + *(_QWORD *)(v26 + 24));
  v33 = *(_QWORD *)(v26 + 16) - (_QWORD)v27;
  *(_QWORD *)(v26 + 24) = v27;
  if ( v33 > 3 )
  {
LABEL_17:
    *v27 = 544108320;
    v29 = (_BYTE *)(*(_QWORD *)(v26 + 24) + 4LL);
    v30 = *(_BYTE **)(v26 + 16);
    *(_QWORD *)(v26 + 24) = v29;
    if ( v30 - v29 >= a5 )
      goto LABEL_18;
LABEL_26:
    v26 = sub_16E7EE0(v26, a4, a5);
    v30 = *(_BYTE **)(v26 + 16);
    v29 = *(_BYTE **)(v26 + 24);
    goto LABEL_19;
  }
LABEL_25:
  v26 = sub_16E7EE0(v26, " on ", 4);
  v29 = *(_BYTE **)(v26 + 24);
  v30 = *(_BYTE **)(v26 + 16);
  if ( v30 - v29 < a5 )
    goto LABEL_26;
LABEL_18:
  if ( a5 )
  {
    memcpy(v29, a4, a5);
    v32 = *(_BYTE **)(v26 + 16);
    v29 = (_BYTE *)(a5 + *(_QWORD *)(v26 + 24));
    *(_QWORD *)(v26 + 24) = v29;
    if ( v32 != v29 )
      goto LABEL_20;
LABEL_23:
    sub_16E7EE0(v26, "\n", 1);
    return v6;
  }
LABEL_19:
  if ( v30 == v29 )
    goto LABEL_23;
LABEL_20:
  *v29 = 10;
  ++*(_QWORD *)(v26 + 24);
  return v6;
}
