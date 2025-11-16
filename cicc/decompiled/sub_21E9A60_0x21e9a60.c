// Function: sub_21E9A60
// Address: 0x21e9a60
//
__int64 __fastcall sub_21E9A60(__int64 a1, __int64 *a2)
{
  _BYTE *v5; // rax
  _BYTE *v6; // rcx
  _BYTE *v7; // rdx
  char v8; // si
  char v9; // dl
  __int64 v10; // r13
  void *v11; // rdx
  const char *v12; // rax
  size_t v13; // rdx
  _DWORD *v14; // rdi
  char *v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 v24; // r14
  __m128i *v25; // rdx
  __int64 v26; // rax
  _QWORD *v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rdi
  _BYTE *v30; // rax
  __int64 v31; // rax
  __int64 v32; // r9
  void *v33; // rdx
  __int64 *v34; // rsi
  unsigned int v35; // edi
  __int64 *v36; // rax
  int v37; // edx
  __int64 v38; // rsi
  __int64 v39; // rdi
  _BYTE *v40; // rax
  __int64 v41; // rax
  size_t v42; // [rsp+8h] [rbp-28h]

  sub_1E0A440(a2);
  if ( !(unsigned __int8)sub_160E740() )
    return 0;
  v5 = (_BYTE *)qword_4F9E580[20];
  v6 = (_BYTE *)qword_4F9E580[21];
  if ( v5 == v6 )
    goto LABEL_31;
  v7 = (_BYTE *)qword_4F9E580[20];
  v8 = 0;
  do
    v8 |= *v7++;
  while ( v6 != v7 );
  if ( (v8 & 1) == 0 )
    goto LABEL_7;
  v19 = *(__int64 **)(a1 + 8);
  v20 = *v19;
  v21 = v19[1];
  if ( v20 == v21 )
LABEL_54:
    BUG();
  while ( *(_UNKNOWN **)v20 != &unk_4FD4138 )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_54;
  }
  v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(*(_QWORD *)(v20 + 8), &unk_4FD4138);
  v23 = *(_QWORD *)(a1 + 232);
  v24 = *(_QWORD *)(v22 + 232);
  v25 = *(__m128i **)(v23 + 24);
  if ( *(_QWORD *)(v23 + 16) - (_QWORD)v25 <= 0xFu )
  {
    v23 = sub_16E7EE0(v23, "Max Live RRegs: ", 0x10u);
  }
  else
  {
    *v25 = _mm_load_si128((const __m128i *)&xmmword_3F6EF60);
    *(_QWORD *)(v23 + 24) += 16LL;
  }
  v26 = sub_16E7AB0(v23, *(int *)(v24 + 24));
  v27 = *(_QWORD **)(v26 + 24);
  v28 = v26;
  if ( *(_QWORD *)(v26 + 16) - (_QWORD)v27 <= 7u )
  {
    v28 = sub_16E7EE0(v26, "\tPRegs: ", 8u);
  }
  else
  {
    *v27 = 0x203A736765525009LL;
    *(_QWORD *)(v26 + 24) += 8LL;
  }
  v29 = sub_16E7AB0(v28, *(int *)(v24 + 28));
  v30 = *(_BYTE **)(v29 + 24);
  if ( *(_BYTE **)(v29 + 16) == v30 )
  {
    sub_16E7EE0(v29, "\t", 1u);
  }
  else
  {
    *v30 = 9;
    ++*(_QWORD *)(v29 + 24);
  }
  v5 = (_BYTE *)qword_4F9E580[20];
  v6 = (_BYTE *)qword_4F9E580[21];
  if ( v5 == v6 )
  {
LABEL_31:
    v9 = 0;
  }
  else
  {
LABEL_7:
    v9 = 0;
    do
      v9 |= *v5++;
    while ( v6 != v5 );
  }
  if ( (v9 & 2) != 0 )
  {
    v32 = *(_QWORD *)(a1 + 232);
    v33 = *(void **)(v32 + 24);
    if ( *(_QWORD *)(v32 + 16) - (_QWORD)v33 <= 0xEu )
    {
      v32 = sub_16E7EE0(*(_QWORD *)(a1 + 232), "Function Size: ", 0xFu);
    }
    else
    {
      qmemcpy(v33, "Function Size: ", 15);
      *(_QWORD *)(v32 + 24) += 15LL;
    }
    v34 = (__int64 *)a2[41];
    if ( v34 == a2 + 40 )
    {
      v38 = 0;
    }
    else
    {
      v35 = 0;
      do
      {
        v36 = (__int64 *)v34[4];
        if ( v34 + 3 != v36 )
        {
          v37 = 0;
          do
          {
            v36 = (__int64 *)v36[1];
            ++v37;
          }
          while ( v34 + 3 != v36 );
          v35 += v37;
        }
        v34 = (__int64 *)v34[1];
      }
      while ( a2 + 40 != v34 );
      v38 = v35;
    }
    v39 = sub_16E7A90(v32, v38);
    v40 = *(_BYTE **)(v39 + 24);
    if ( *(_BYTE **)(v39 + 16) == v40 )
    {
      sub_16E7EE0(v39, "\t", 1u);
    }
    else
    {
      *v40 = 9;
      ++*(_QWORD *)(v39 + 24);
    }
  }
  v10 = *(_QWORD *)(a1 + 232);
  v11 = *(void **)(v10 + 24);
  if ( *(_QWORD *)(v10 + 16) - (_QWORD)v11 <= 9u )
  {
    v10 = sub_16E7EE0(*(_QWORD *)(a1 + 232), "Function: ", 0xAu);
  }
  else
  {
    qmemcpy(v11, "Function: ", 10);
    *(_QWORD *)(v10 + 24) += 10LL;
  }
  v12 = sub_1E0A440(a2);
  v14 = *(_DWORD **)(v10 + 24);
  v15 = (char *)v12;
  v16 = *(_QWORD *)(v10 + 16) - (_QWORD)v14;
  if ( v13 > v16 )
  {
    v31 = sub_16E7EE0(v10, v15, v13);
    v14 = *(_DWORD **)(v31 + 24);
    v10 = v31;
    v16 = *(_QWORD *)(v31 + 16) - (_QWORD)v14;
  }
  else if ( v13 )
  {
    v42 = v13;
    memcpy(v14, v15, v13);
    v41 = *(_QWORD *)(v10 + 16);
    v14 = (_DWORD *)(v42 + *(_QWORD *)(v10 + 24));
    *(_QWORD *)(v10 + 24) = v14;
    v16 = v41 - (_QWORD)v14;
  }
  if ( v16 <= 6 )
  {
    v10 = sub_16E7EE0(v10, "\tPass: ", 7u);
  }
  else
  {
    *v14 = 1935757321;
    *((_WORD *)v14 + 2) = 14963;
    *((_BYTE *)v14 + 6) = 32;
    *(_QWORD *)(v10 + 24) += 7LL;
  }
  v17 = sub_16E7EE0(v10, *(char **)(a1 + 240), *(_QWORD *)(a1 + 248));
  v18 = *(_BYTE **)(v17 + 24);
  if ( *(_BYTE **)(v17 + 16) == v18 )
  {
    sub_16E7EE0(v17, "\n", 1u);
  }
  else
  {
    *v18 = 10;
    ++*(_QWORD *)(v17 + 24);
  }
  return 0;
}
