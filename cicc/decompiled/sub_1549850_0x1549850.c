// Function: sub_1549850
// Address: 0x1549850
//
void *__fastcall sub_1549850(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  _BYTE *v4; // rdi
  unsigned __int64 v5; // rax
  const char *v6; // rax
  size_t v7; // rdx
  size_t v8; // r12
  __int64 v9; // r13
  void *v10; // rdi
  void *result; // rax
  const char *v12; // r15
  size_t v13; // rax
  size_t v14; // r14
  unsigned __int64 v15; // rax

  v3 = *a1;
  if ( *((_BYTE *)a1 + 8) )
  {
    *((_BYTE *)a1 + 8) = 0;
    v4 = *(_BYTE **)(v3 + 24);
    v5 = *(_QWORD *)(v3 + 16) - (_QWORD)v4;
    goto LABEL_3;
  }
  v12 = (const char *)a1[2];
  if ( !v12 )
    goto LABEL_15;
  v13 = strlen((const char *)a1[2]);
  v4 = *(_BYTE **)(v3 + 24);
  v14 = v13;
  v5 = *(_QWORD *)(v3 + 16) - (_QWORD)v4;
  if ( v14 > v5 )
  {
    v3 = sub_16E7EE0(v3, v12, v14);
LABEL_15:
    v4 = *(_BYTE **)(v3 + 24);
    v5 = *(_QWORD *)(v3 + 16) - (_QWORD)v4;
    goto LABEL_3;
  }
  if ( v14 )
  {
    memcpy(v4, v12, v14);
    v4 = (_BYTE *)(v14 + *(_QWORD *)(v3 + 24));
    v15 = *(_QWORD *)(v3 + 16) - (_QWORD)v4;
    *(_QWORD *)(v3 + 24) = v4;
    if ( v15 > 4 )
      goto LABEL_4;
    goto LABEL_13;
  }
LABEL_3:
  if ( v5 > 4 )
  {
LABEL_4:
    *(_DWORD *)v4 = 979853684;
    v4[4] = 32;
    *(_QWORD *)(v3 + 24) += 5LL;
    goto LABEL_5;
  }
LABEL_13:
  sub_16E7EE0(v3, "tag: ", 5);
LABEL_5:
  v6 = sub_14E0540(*(unsigned __int16 *)(a2 + 2));
  v8 = v7;
  if ( !v7 )
    return (void *)sub_16E7A90(*a1, *(unsigned __int16 *)(a2 + 2));
  v9 = *a1;
  v10 = *(void **)(*a1 + 24);
  if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v10 < v7 )
    return (void *)sub_16E7EE0(*a1, v6, v7);
  result = memcpy(v10, v6, v7);
  *(_QWORD *)(v9 + 24) += v8;
  return result;
}
