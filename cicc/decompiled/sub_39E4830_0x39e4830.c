// Function: sub_39E4830
// Address: 0x39e4830
//
_BYTE *__fastcall sub_39E4830(__int64 a1, _BYTE *a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rdi
  _QWORD *v8; // rdx
  __int64 v9; // rdi
  _BYTE *v10; // rax
  int v11; // eax
  unsigned __int64 v12; // r13
  _BYTE *result; // rax
  __int64 v14; // rdi
  __int64 v15; // r14
  char *v16; // rsi
  size_t v17; // rdx
  void *v18; // rdi
  __int64 v19; // rdi
  _BYTE *v20; // rax
  __int64 v21; // rdi
  _BYTE *v22; // rax

  v6 = *(_QWORD *)(a1 + 272);
  v8 = *(_QWORD **)(v6 + 24);
  if ( *(_QWORD *)(v6 + 16) - (_QWORD)v8 <= 7u )
  {
    sub_16E7EE0(v6, "\t.lcomm\t", 8u);
  }
  else
  {
    *v8 = 0x96D6D6F636C2E09LL;
    *(_QWORD *)(v6 + 24) += 8LL;
  }
  sub_38E2490(a2, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
  v9 = *(_QWORD *)(a1 + 272);
  v10 = *(_BYTE **)(v9 + 24);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 16) )
  {
    v9 = sub_16E7DE0(v9, 44);
  }
  else
  {
    *(_QWORD *)(v9 + 24) = v10 + 1;
    *v10 = 44;
  }
  sub_16E7A90(v9, a3);
  if ( a4 > 1 )
  {
    v11 = *(_DWORD *)(*(_QWORD *)(a1 + 280) + 300LL);
    if ( v11 == 1 )
    {
      v21 = *(_QWORD *)(a1 + 272);
      v22 = *(_BYTE **)(v21 + 24);
      if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 16) )
      {
        v21 = sub_16E7DE0(v21, 44);
      }
      else
      {
        *(_QWORD *)(v21 + 24) = v22 + 1;
        *v22 = 44;
      }
      sub_16E7A90(v21, a4);
    }
    else if ( v11 == 2 )
    {
      v19 = *(_QWORD *)(a1 + 272);
      v20 = *(_BYTE **)(v19 + 24);
      if ( (unsigned __int64)v20 >= *(_QWORD *)(v19 + 16) )
      {
        v19 = sub_16E7DE0(v19, 44);
      }
      else
      {
        *(_QWORD *)(v19 + 24) = v20 + 1;
        *v20 = 44;
      }
      _BitScanReverse(&a4, a4);
      sub_16E7A90(v19, 31 - (a4 ^ 0x1F));
    }
  }
  v12 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v15 = *(_QWORD *)(a1 + 272);
    v16 = *(char **)(a1 + 304);
    v17 = *(unsigned int *)(a1 + 312);
    v18 = *(void **)(v15 + 24);
    if ( v12 > *(_QWORD *)(v15 + 16) - (_QWORD)v18 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v16, v17);
    }
    else
    {
      memcpy(v18, v16, v17);
      *(_QWORD *)(v15 + 24) += v12;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v14 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v14 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v14 + 16) )
    return (_BYTE *)sub_16E7DE0(v14, 10);
  *(_QWORD *)(v14 + 24) = result + 1;
  *result = 10;
  return result;
}
