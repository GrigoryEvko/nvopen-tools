// Function: sub_38CDA50
// Address: 0x38cda50
//
void *__fastcall sub_38CDA50(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // rax
  unsigned __int64 v4; // rdx
  char *v5; // rax
  size_t v6; // rdx
  _BYTE *v7; // rdi
  char *v8; // rsi
  unsigned __int64 v9; // rax
  size_t v10; // r13
  void *result; // rax
  char *v12; // rax
  size_t v13; // rdx
  void *v14; // rdi
  char *v15; // rsi
  __int64 v16; // rax
  size_t v17; // [rsp+8h] [rbp-28h]

  v2 = a2;
  v3 = *(_BYTE **)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 16);
  if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
  {
    if ( v4 <= (unsigned __int64)v3 )
    {
      v2 = sub_16E7DE0(a2, 40);
    }
    else
    {
      *(_QWORD *)(a2 + 24) = v3 + 1;
      *v3 = 40;
    }
    v5 = sub_38CB5D0(*(_WORD *)(a1 + 16));
    v7 = *(_BYTE **)(v2 + 24);
    v8 = v5;
    v9 = *(_QWORD *)(v2 + 16);
    v10 = v6;
    if ( v9 - (unsigned __int64)v7 < v6 )
    {
      v16 = sub_16E7EE0(v2, v8, v6);
      v7 = *(_BYTE **)(v16 + 24);
      v2 = v16;
      if ( (unsigned __int64)v7 < *(_QWORD *)(v16 + 16) )
      {
LABEL_8:
        result = v7 + 1;
        *(_QWORD *)(v2 + 24) = v7 + 1;
        *v7 = 41;
        return result;
      }
    }
    else
    {
      if ( v6 )
      {
        memcpy(v7, v8, v6);
        v9 = *(_QWORD *)(v2 + 16);
        v7 = (_BYTE *)(v10 + *(_QWORD *)(v2 + 24));
        *(_QWORD *)(v2 + 24) = v7;
      }
      if ( (unsigned __int64)v7 < v9 )
        goto LABEL_8;
    }
    return (void *)sub_16E7DE0(v2, 41);
  }
  if ( v4 <= (unsigned __int64)v3 )
  {
    v2 = sub_16E7DE0(a2, 64);
  }
  else
  {
    *(_QWORD *)(a2 + 24) = v3 + 1;
    *v3 = 64;
  }
  v12 = sub_38CB5D0(*(_WORD *)(a1 + 16));
  v14 = *(void **)(v2 + 24);
  v15 = v12;
  result = (void *)(*(_QWORD *)(v2 + 16) - (_QWORD)v14);
  if ( (unsigned __int64)result < v13 )
    return (void *)sub_16E7EE0(v2, v15, v13);
  if ( v13 )
  {
    v17 = v13;
    result = memcpy(v14, v15, v13);
    *(_QWORD *)(v2 + 24) += v17;
  }
  return result;
}
