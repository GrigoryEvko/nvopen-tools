// Function: sub_39E5090
// Address: 0x39e5090
//
_BYTE *__fastcall sub_39E5090(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4, unsigned int a5)
{
  unsigned int v5; // r9d
  __int64 v10; // r12
  _BYTE *v11; // rax
  char *v12; // rsi
  size_t v13; // rax
  _QWORD *v14; // rcx
  size_t v15; // rdx
  _BYTE *v16; // rdx
  __int64 v17; // rax
  _WORD *v18; // rdx
  __int64 v19; // rdi
  unsigned __int64 v20; // r12
  _BYTE *result; // rax
  unsigned __int64 v22; // r8
  char *v23; // rcx
  char *v24; // rsi
  unsigned int v25; // ecx
  unsigned int v26; // ecx
  unsigned int v27; // edi
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r14
  char *v32; // rsi
  size_t v33; // rdx
  void *v34; // rdi
  __int64 v35; // rdi
  _WORD *v36; // rdx
  __int64 v37; // rax

  v5 = a2;
  v10 = *(_QWORD *)(a1 + 272);
  v11 = *(_BYTE **)(v10 + 24);
  if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 16) )
  {
    v30 = sub_16E7DE0(v10, 9);
    v5 = a2;
    v10 = v30;
  }
  else
  {
    *(_QWORD *)(v10 + 24) = v11 + 1;
    *v11 = 9;
  }
  v12 = ".macosx_version_min";
  if ( v5 != 1 )
  {
    v12 = ".ios_version_min";
    if ( v5 > 1 )
    {
      v12 = ".watchos_version_min";
      if ( v5 == 2 )
        v12 = ".tvos_version_min";
    }
  }
  v13 = strlen(v12);
  v14 = *(_QWORD **)(v10 + 24);
  v15 = v13;
  if ( v13 <= *(_QWORD *)(v10 + 16) - (_QWORD)v14 )
  {
    v22 = (unsigned __int64)(v14 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v14 = *(_QWORD *)v12;
    *(_QWORD *)((char *)v14 + (unsigned int)v13 - 8) = *(_QWORD *)&v12[(unsigned int)v13 - 8];
    v23 = (char *)v14 - v22;
    v24 = (char *)(v12 - v23);
    v25 = (v13 + (_DWORD)v23) & 0xFFFFFFF8;
    if ( v25 >= 8 )
    {
      v26 = v25 & 0xFFFFFFF8;
      v27 = 0;
      do
      {
        v28 = v27;
        v27 += 8;
        *(_QWORD *)(v22 + v28) = *(_QWORD *)&v24[v28];
      }
      while ( v27 < v26 );
    }
    v16 = (_BYTE *)(*(_QWORD *)(v10 + 24) + v15);
    *(_QWORD *)(v10 + 24) = v16;
    if ( *(_QWORD *)(v10 + 16) > (unsigned __int64)v16 )
      goto LABEL_9;
  }
  else
  {
    v10 = sub_16E7EE0(v10, v12, v13);
    v16 = *(_BYTE **)(v10 + 24);
    if ( *(_QWORD *)(v10 + 16) > (unsigned __int64)v16 )
    {
LABEL_9:
      *(_QWORD *)(v10 + 24) = v16 + 1;
      *v16 = 32;
      goto LABEL_10;
    }
  }
  v10 = sub_16E7DE0(v10, 32);
LABEL_10:
  v17 = sub_16E7A90(v10, a3);
  v18 = *(_WORD **)(v17 + 24);
  v19 = v17;
  if ( *(_QWORD *)(v17 + 16) - (_QWORD)v18 <= 1u )
  {
    v19 = sub_16E7EE0(v17, ", ", 2u);
  }
  else
  {
    *v18 = 8236;
    *(_QWORD *)(v17 + 24) += 2LL;
  }
  sub_16E7A90(v19, a4);
  if ( a5 )
  {
    v35 = *(_QWORD *)(a1 + 272);
    v36 = *(_WORD **)(v35 + 24);
    if ( *(_QWORD *)(v35 + 16) - (_QWORD)v36 <= 1u )
    {
      v37 = sub_16E7EE0(v35, ", ", 2u);
      sub_16E7A90(v37, a5);
    }
    else
    {
      *v36 = 8236;
      *(_QWORD *)(v35 + 24) += 2LL;
      sub_16E7A90(v35, a5);
    }
  }
  v20 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v31 = *(_QWORD *)(a1 + 272);
    v32 = *(char **)(a1 + 304);
    v33 = *(unsigned int *)(a1 + 312);
    v34 = *(void **)(v31 + 24);
    if ( v20 > *(_QWORD *)(v31 + 16) - (_QWORD)v34 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v32, v33);
    }
    else
    {
      memcpy(v34, v32, v33);
      *(_QWORD *)(v31 + 24) += v20;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v29 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v29 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v29 + 16) )
    return (_BYTE *)sub_16E7DE0(v29, 10);
  *(_QWORD *)(v29 + 24) = result + 1;
  *result = 10;
  return result;
}
