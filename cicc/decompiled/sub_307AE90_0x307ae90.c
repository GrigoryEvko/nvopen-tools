// Function: sub_307AE90
// Address: 0x307ae90
//
__int64 __fastcall sub_307AE90(__int64 a1, __int64 *a2)
{
  char *v4; // rax
  __int64 v5; // rdx
  _BYTE *v7; // rax
  _BYTE *v8; // rcx
  _BYTE *v9; // rdx
  char v10; // si
  char v11; // dl
  __int64 v12; // r13
  void *v13; // rdx
  const char *v14; // rax
  size_t v15; // rdx
  _DWORD *v16; // rdi
  unsigned __int8 *v17; // rsi
  unsigned __int64 v18; // rax
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // r14
  __m128i *v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rdi
  _BYTE *v32; // rax
  __int64 v33; // rax
  __int64 v34; // r9
  void *v35; // rdx
  __int64 *v36; // rsi
  unsigned int v37; // edi
  __int64 *v38; // rax
  int v39; // edx
  unsigned __int64 v40; // rsi
  __int64 v41; // rdi
  _BYTE *v42; // rax
  __int64 v43; // rax
  size_t v44; // [rsp+8h] [rbp-28h]

  v4 = (char *)sub_2E791E0(a2);
  if ( !sub_BC63A0(v4, v5) )
    return 0;
  v7 = (_BYTE *)unk_4F83008;
  v8 = (_BYTE *)unk_4F83010;
  if ( unk_4F83008 == unk_4F83010 )
    goto LABEL_31;
  v9 = (_BYTE *)unk_4F83008;
  v10 = 0;
  do
    v10 |= *v9++;
  while ( (_BYTE *)unk_4F83010 != v9 );
  if ( (v10 & 1) == 0 )
    goto LABEL_7;
  v21 = *(__int64 **)(a1 + 8);
  v22 = *v21;
  v23 = v21[1];
  if ( v22 == v23 )
LABEL_54:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_502D274 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_54;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_502D274);
  v25 = *(_QWORD *)(a1 + 200);
  v26 = *(_QWORD *)(v24 + 200);
  v27 = *(__m128i **)(v25 + 32);
  if ( *(_QWORD *)(v25 + 24) - (_QWORD)v27 <= 0xFu )
  {
    v25 = sub_CB6200(v25, "Max Live RRegs: ", 0x10u);
  }
  else
  {
    *v27 = _mm_load_si128((const __m128i *)&xmmword_3F6EF60);
    *(_QWORD *)(v25 + 32) += 16LL;
  }
  v28 = sub_CB59F0(v25, *(int *)(v26 + 24));
  v29 = *(_QWORD **)(v28 + 32);
  v30 = v28;
  if ( *(_QWORD *)(v28 + 24) - (_QWORD)v29 <= 7u )
  {
    v30 = sub_CB6200(v28, "\tPRegs: ", 8u);
  }
  else
  {
    *v29 = 0x203A736765525009LL;
    *(_QWORD *)(v28 + 32) += 8LL;
  }
  v31 = sub_CB59F0(v30, *(int *)(v26 + 28));
  v32 = *(_BYTE **)(v31 + 32);
  if ( *(_BYTE **)(v31 + 24) == v32 )
  {
    sub_CB6200(v31, (unsigned __int8 *)"\t", 1u);
  }
  else
  {
    *v32 = 9;
    ++*(_QWORD *)(v31 + 32);
  }
  v7 = (_BYTE *)unk_4F83008;
  v8 = (_BYTE *)unk_4F83010;
  if ( unk_4F83008 == unk_4F83010 )
  {
LABEL_31:
    v11 = 0;
  }
  else
  {
LABEL_7:
    v11 = 0;
    do
      v11 |= *v7++;
    while ( v8 != v7 );
  }
  if ( (v11 & 2) != 0 )
  {
    v34 = *(_QWORD *)(a1 + 200);
    v35 = *(void **)(v34 + 32);
    if ( *(_QWORD *)(v34 + 24) - (_QWORD)v35 <= 0xEu )
    {
      v34 = sub_CB6200(*(_QWORD *)(a1 + 200), "Function Size: ", 0xFu);
    }
    else
    {
      qmemcpy(v35, "Function Size: ", 15);
      *(_QWORD *)(v34 + 32) += 15LL;
    }
    v36 = (__int64 *)a2[41];
    if ( v36 == a2 + 40 )
    {
      v40 = 0;
    }
    else
    {
      v37 = 0;
      do
      {
        v38 = (__int64 *)v36[7];
        if ( v36 + 6 != v38 )
        {
          v39 = 0;
          do
          {
            v38 = (__int64 *)v38[1];
            ++v39;
          }
          while ( v36 + 6 != v38 );
          v37 += v39;
        }
        v36 = (__int64 *)v36[1];
      }
      while ( a2 + 40 != v36 );
      v40 = v37;
    }
    v41 = sub_CB59D0(v34, v40);
    v42 = *(_BYTE **)(v41 + 32);
    if ( *(_BYTE **)(v41 + 24) == v42 )
    {
      sub_CB6200(v41, (unsigned __int8 *)"\t", 1u);
    }
    else
    {
      *v42 = 9;
      ++*(_QWORD *)(v41 + 32);
    }
  }
  v12 = *(_QWORD *)(a1 + 200);
  v13 = *(void **)(v12 + 32);
  if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 9u )
  {
    v12 = sub_CB6200(*(_QWORD *)(a1 + 200), (unsigned __int8 *)"Function: ", 0xAu);
  }
  else
  {
    qmemcpy(v13, "Function: ", 10);
    *(_QWORD *)(v12 + 32) += 10LL;
  }
  v14 = sub_2E791E0(a2);
  v16 = *(_DWORD **)(v12 + 32);
  v17 = (unsigned __int8 *)v14;
  v18 = *(_QWORD *)(v12 + 24) - (_QWORD)v16;
  if ( v15 > v18 )
  {
    v33 = sub_CB6200(v12, v17, v15);
    v16 = *(_DWORD **)(v33 + 32);
    v12 = v33;
    v18 = *(_QWORD *)(v33 + 24) - (_QWORD)v16;
  }
  else if ( v15 )
  {
    v44 = v15;
    memcpy(v16, v17, v15);
    v43 = *(_QWORD *)(v12 + 24);
    v16 = (_DWORD *)(v44 + *(_QWORD *)(v12 + 32));
    *(_QWORD *)(v12 + 32) = v16;
    v18 = v43 - (_QWORD)v16;
  }
  if ( v18 <= 6 )
  {
    v12 = sub_CB6200(v12, "\tPass: ", 7u);
  }
  else
  {
    *v16 = 1935757321;
    *((_WORD *)v16 + 2) = 14963;
    *((_BYTE *)v16 + 6) = 32;
    *(_QWORD *)(v12 + 32) += 7LL;
  }
  v19 = sub_CB6200(v12, *(unsigned __int8 **)(a1 + 208), *(_QWORD *)(a1 + 216));
  v20 = *(_BYTE **)(v19 + 32);
  if ( *(_BYTE **)(v19 + 24) == v20 )
  {
    sub_CB6200(v19, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v20 = 10;
    ++*(_QWORD *)(v19 + 32);
  }
  return 0;
}
