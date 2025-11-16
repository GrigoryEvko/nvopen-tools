// Function: sub_39E2690
// Address: 0x39e2690
//
_BYTE *__fastcall sub_39E2690(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  size_t v5; // rax
  void *v6; // rdi
  size_t v7; // r15
  unsigned __int64 v8; // r13
  _BYTE *result; // rax
  __int64 v10; // rdi
  __int64 v11; // r14
  char *v12; // rsi
  size_t v13; // rdx
  void *v14; // rdi
  char *src; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 280);
  v4 = *(_QWORD *)(a1 + 272);
  if ( *(_QWORD *)(v3 + 256) )
  {
    src = *(char **)(v3 + 256);
    v5 = strlen(src);
    v6 = *(void **)(v4 + 24);
    v7 = v5;
    if ( v5 > *(_QWORD *)(v4 + 16) - (_QWORD)v6 )
    {
      sub_16E7EE0(v4, src, v5);
      v3 = *(_QWORD *)(a1 + 280);
      v4 = *(_QWORD *)(a1 + 272);
    }
    else if ( v5 )
    {
      memcpy(v6, src, v5);
      *(_QWORD *)(v4 + 24) += v7;
      v3 = *(_QWORD *)(a1 + 280);
      v4 = *(_QWORD *)(a1 + 272);
    }
  }
  sub_38CDBE0(a2, v4, v3);
  v8 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v11 = *(_QWORD *)(a1 + 272);
    v12 = *(char **)(a1 + 304);
    v13 = *(unsigned int *)(a1 + 312);
    v14 = *(void **)(v11 + 24);
    if ( v8 > *(_QWORD *)(v11 + 16) - (_QWORD)v14 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v12, v13);
    }
    else
    {
      memcpy(v14, v12, v13);
      *(_QWORD *)(v11 + 24) += v8;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v10 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v10 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v10 + 16) )
    return (_BYTE *)sub_16E7DE0(v10, 10);
  *(_QWORD *)(v10 + 24) = result + 1;
  *result = 10;
  return result;
}
