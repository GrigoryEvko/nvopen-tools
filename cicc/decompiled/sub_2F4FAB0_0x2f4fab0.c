// Function: sub_2F4FAB0
// Address: 0x2f4fab0
//
__int64 __fastcall sub_2F4FAB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  size_t v3; // r13
  char *v4; // r14
  __int64 v5; // rdx
  void *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  _BYTE *v9; // r13
  unsigned __int64 v11; // rax

  v2 = a2;
  v3 = *(_QWORD *)(a1 + 40);
  if ( v3 )
  {
    v4 = *(char **)(a1 + 32);
  }
  else
  {
    v3 = 3;
    v4 = "all";
  }
  v5 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v5) > 6 )
  {
    *(_DWORD *)v5 = 1701147239;
    *(_WORD *)(v5 + 4) = 31076;
    *(_BYTE *)(v5 + 6) = 60;
    v6 = (void *)(*(_QWORD *)(a2 + 32) + 7LL);
    v7 = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 32) = v6;
    if ( v3 > v7 - (__int64)v6 )
      goto LABEL_5;
LABEL_8:
    memcpy(v6, v4, v3);
    v11 = *(_QWORD *)(v2 + 24);
    v9 = (_BYTE *)(*(_QWORD *)(v2 + 32) + v3);
    *(_QWORD *)(v2 + 32) = v9;
    if ( (unsigned __int64)v9 < v11 )
      goto LABEL_6;
    return sub_CB5D20(v2, 62);
  }
  v2 = sub_CB6200(a2, "greedy<", 7u);
  v6 = *(void **)(v2 + 32);
  if ( v3 <= *(_QWORD *)(v2 + 24) - (_QWORD)v6 )
    goto LABEL_8;
LABEL_5:
  v8 = sub_CB6200(v2, (unsigned __int8 *)v4, v3);
  v9 = *(_BYTE **)(v8 + 32);
  v2 = v8;
  if ( (unsigned __int64)v9 < *(_QWORD *)(v8 + 24) )
  {
LABEL_6:
    *(_QWORD *)(v2 + 32) = v9 + 1;
    *v9 = 62;
    return (__int64)(v9 + 1);
  }
  return sub_CB5D20(v2, 62);
}
