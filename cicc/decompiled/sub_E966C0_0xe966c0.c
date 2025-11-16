// Function: sub_E966C0
// Address: 0xe966c0
//
_BYTE *__fastcall sub_E966C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  void *v8; // rdx
  _WORD *v9; // rdx
  _BYTE *v10; // rax
  int v11; // edx
  _BYTE *v12; // rax
  _BYTE *v13; // rax
  unsigned __int64 v14; // rcx
  _BYTE *result; // rax
  __int64 v16; // rax
  size_t *v17; // rsi
  size_t v18; // rdx
  void *v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __int64 v23; // r14
  void *v24; // rdi
  unsigned __int64 v25; // r15
  unsigned __int8 *v26; // rsi
  __int64 v27; // rdi
  void *v28; // rdx
  _BYTE *v29; // rax
  __int64 v30; // rdi

  if ( (unsigned __int8)sub_E966B0(a1, *(_QWORD *)(a1 + 128), *(_QWORD *)(a1 + 136), a2) )
  {
    v22 = *(_BYTE **)(a4 + 32);
    if ( (unsigned __int64)v22 >= *(_QWORD *)(a4 + 24) )
    {
      v23 = sub_CB5D20(a4, 9);
    }
    else
    {
      v23 = a4;
      *(_QWORD *)(a4 + 32) = v22 + 1;
      *v22 = 9;
    }
    v24 = *(void **)(v23 + 32);
    v25 = *(_QWORD *)(a1 + 136);
    v26 = *(unsigned __int8 **)(a1 + 128);
    if ( v25 > *(_QWORD *)(v23 + 24) - (_QWORD)v24 )
    {
      sub_CB6200(v23, v26, *(_QWORD *)(a1 + 136));
    }
    else if ( v25 )
    {
      memcpy(v24, v26, *(_QWORD *)(a1 + 136));
      *(_QWORD *)(v23 + 32) += v25;
    }
    if ( a5 )
    {
      v29 = *(_BYTE **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v29 )
      {
        v30 = sub_CB5D20(a4, 9);
      }
      else
      {
        v30 = a4;
        *(_QWORD *)(a4 + 32) = v29 + 1;
        *v29 = 9;
      }
      sub_CB59D0(v30, a5);
    }
    result = *(_BYTE **)(a4 + 32);
    v27 = a4;
    if ( (unsigned __int64)result < *(_QWORD *)(a4 + 24) )
    {
      *(_QWORD *)(a4 + 32) = result + 1;
      *result = 10;
      return result;
    }
    return (_BYTE *)sub_CB5D20(v27, 10);
  }
  v8 = *(void **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v8 <= 9u )
  {
    sub_CB6200(a4, "\t.section\t", 0xAu);
  }
  else
  {
    qmemcpy(v8, "\t.section\t", 10);
    *(_QWORD *)(a4 + 32) += 10LL;
  }
  sub_E96430(a4, *(void **)(a1 + 128), *(_QWORD *)(a1 + 136));
  v9 = *(_WORD **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v9 <= 1u )
  {
    sub_CB6200(a4, (unsigned __int8 *)",\"", 2u);
    v10 = *(_BYTE **)(a4 + 32);
    if ( !*(_BYTE *)(a1 + 172) )
      goto LABEL_7;
  }
  else
  {
    *v9 = 8748;
    v10 = (_BYTE *)(*(_QWORD *)(a4 + 32) + 2LL);
    *(_QWORD *)(a4 + 32) = v10;
    if ( !*(_BYTE *)(a1 + 172) )
      goto LABEL_7;
  }
  if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v10 )
  {
    sub_CB5D20(a4, 112);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v10 + 1;
    *v10 = 112;
  }
  v10 = *(_BYTE **)(a4 + 32);
LABEL_7:
  if ( *(_QWORD *)(a1 + 152) )
  {
    if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v10 )
    {
      sub_CB5D20(a4, 71);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v10 + 1;
      *v10 = 71;
    }
    v10 = *(_BYTE **)(a4 + 32);
  }
  v11 = *(_DWORD *)(a1 + 176);
  if ( (v11 & 1) != 0 )
  {
    if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v10 )
    {
      sub_CB5D20(a4, 83);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v10 + 1;
      *v10 = 83;
    }
    v11 = *(_DWORD *)(a1 + 176);
    v10 = *(_BYTE **)(a4 + 32);
  }
  if ( (v11 & 2) != 0 )
  {
    if ( (unsigned __int64)v10 >= *(_QWORD *)(a4 + 24) )
    {
      sub_CB5D20(a4, 84);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v10 + 1;
      *v10 = 84;
    }
    v11 = *(_DWORD *)(a1 + 176);
    v10 = *(_BYTE **)(a4 + 32);
  }
  if ( (v11 & 4) != 0 )
  {
    if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v10 )
    {
      sub_CB5D20(a4, 82);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v10 + 1;
      *v10 = 82;
    }
    v10 = *(_BYTE **)(a4 + 32);
  }
  if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v10 )
  {
    sub_CB5D20(a4, 34);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v10 + 1;
    *v10 = 34;
  }
  v12 = *(_BYTE **)(a4 + 32);
  if ( (unsigned __int64)v12 >= *(_QWORD *)(a4 + 24) )
  {
    sub_CB5D20(a4, 44);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v12 + 1;
    *v12 = 44;
  }
  v13 = *(_BYTE **)(a4 + 32);
  v14 = *(_QWORD *)(a4 + 24);
  if ( **(_BYTE **)(a2 + 48) == 64 )
  {
    if ( v14 <= (unsigned __int64)v13 )
    {
      sub_CB5D20(a4, 37);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v13 + 1;
      *v13 = 37;
    }
  }
  else if ( v14 <= (unsigned __int64)v13 )
  {
    sub_CB5D20(a4, 64);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v13 + 1;
    *v13 = 64;
  }
  result = *(_BYTE **)(a4 + 32);
  if ( *(_QWORD *)(a1 + 152) )
  {
    if ( result == *(_BYTE **)(a4 + 24) )
    {
      sub_CB6200(a4, (unsigned __int8 *)",", 1u);
      v16 = *(_QWORD *)(a1 + 152);
      if ( (*(_BYTE *)(v16 + 8) & 1) != 0 )
        goto LABEL_24;
    }
    else
    {
      *result = 44;
      ++*(_QWORD *)(a4 + 32);
      v16 = *(_QWORD *)(a1 + 152);
      if ( (*(_BYTE *)(v16 + 8) & 1) != 0 )
      {
LABEL_24:
        v17 = *(size_t **)(v16 - 8);
        v18 = *v17;
        v19 = v17 + 3;
        goto LABEL_25;
      }
    }
    v18 = 0;
    v19 = 0;
LABEL_25:
    sub_E96430(a4, v19, v18);
    v20 = *(_QWORD *)(a4 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v20) <= 6 )
    {
      sub_CB6200(a4, ",comdat", 7u);
      result = *(_BYTE **)(a4 + 32);
    }
    else
    {
      *(_DWORD *)v20 = 1836016428;
      *(_WORD *)(v20 + 4) = 24932;
      *(_BYTE *)(v20 + 6) = 116;
      result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 7LL);
      *(_QWORD *)(a4 + 32) = result;
    }
  }
  if ( *(_DWORD *)(a1 + 148) != -1 )
  {
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)result <= 7u )
    {
      v21 = sub_CB6200(a4, ",unique,", 8u);
    }
    else
    {
      v21 = a4;
      *(_QWORD *)result = 0x2C657571696E752CLL;
      *(_QWORD *)(a4 + 32) += 8LL;
    }
    sub_CB59D0(v21, *(unsigned int *)(a1 + 148));
    result = *(_BYTE **)(a4 + 32);
  }
  if ( (unsigned __int64)result >= *(_QWORD *)(a4 + 24) )
  {
    result = (_BYTE *)sub_CB5D20(a4, 10);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = result + 1;
    *result = 10;
  }
  if ( a5 )
  {
    v28 = *(void **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v28 <= 0xCu )
    {
      a4 = sub_CB6200(a4, "\t.subsection\t", 0xDu);
    }
    else
    {
      qmemcpy(v28, "\t.subsection\t", 13);
      *(_QWORD *)(a4 + 32) += 13LL;
    }
    v27 = sub_CB59D0(a4, a5);
    result = *(_BYTE **)(v27 + 32);
    if ( (unsigned __int64)result < *(_QWORD *)(v27 + 24) )
    {
      *(_QWORD *)(v27 + 32) = result + 1;
      *result = 10;
      return result;
    }
    return (_BYTE *)sub_CB5D20(v27, 10);
  }
  return result;
}
