// Function: sub_250F6E0
// Address: 0x250f6e0
//
__int64 __fastcall sub_250F6E0(__int64 a1, _QWORD *a2)
{
  unsigned __int64 v4; // r14
  _BYTE *v5; // rax
  __int64 v6; // r13
  char v7; // al
  __int64 v8; // rax
  __int64 v9; // r13
  unsigned __int8 *v10; // rax
  size_t v11; // rdx
  void *v12; // rdi
  __int64 v13; // r13
  unsigned __int64 v14; // rdi
  unsigned __int8 *v15; // rax
  size_t v16; // rdx
  void *v17; // rdi
  __int64 v18; // r13
  char v19; // al
  signed __int64 v20; // rsi
  __int64 v21; // rax
  _BYTE *v22; // r13
  _BYTE *v23; // rax
  size_t v25; // [rsp+8h] [rbp-28h]
  size_t v26; // [rsp+8h] [rbp-28h]

  v4 = sub_250D070(a2);
  v5 = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) == v5 )
  {
    v6 = sub_CB6200(a1, (unsigned __int8 *)"{", 1u);
  }
  else
  {
    *v5 = 123;
    v6 = a1;
    ++*(_QWORD *)(a1 + 32);
  }
  v7 = sub_2509800(a2);
  v8 = sub_250F640(v6, v7);
  v9 = sub_904010(v8, ":");
  v10 = (unsigned __int8 *)sub_BD5D20(v4);
  v12 = *(void **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v12 < v11 )
  {
    v9 = sub_CB6200(v9, v10, v11);
  }
  else if ( v11 )
  {
    v26 = v11;
    memcpy(v12, v10, v11);
    *(_QWORD *)(v9 + 32) += v26;
  }
  v13 = sub_904010(v9, " [");
  v14 = *a2 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*a2 & 3LL) == 3 )
    v14 = *(_QWORD *)(v14 + 24);
  v15 = (unsigned __int8 *)sub_BD5D20(v14);
  v17 = *(void **)(v13 + 32);
  if ( v16 > *(_QWORD *)(v13 + 24) - (_QWORD)v17 )
  {
    v13 = sub_CB6200(v13, v15, v16);
  }
  else if ( v16 )
  {
    v25 = v16;
    memcpy(v17, v15, v16);
    *(_QWORD *)(v13 + 32) += v25;
  }
  v18 = sub_904010(v13, "@");
  v19 = sub_2509800(a2);
  if ( v19 == 6 )
  {
    v20 = *(int *)((*a2 & 0xFFFFFFFFFFFFFFFCLL) + 32);
  }
  else
  {
    v20 = -1;
    if ( v19 == 7 )
      v20 = (int)((__int64)((*a2 & 0xFFFFFFFFFFFFFFFCLL)
                          - (*(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFFCLL) + 24)
                           - 32LL * (*(_DWORD *)(*(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFFCLL) + 24) + 4LL) & 0x7FFFFFF))) >> 5);
  }
  v21 = sub_CB59F0(v18, v20);
  sub_904010(v21, "]");
  if ( a2[1] )
  {
    v22 = (_BYTE *)sub_904010(a1, "[cb_context:");
    sub_A69870(a2[1], v22, 0);
    sub_904010((__int64)v22, "]");
  }
  v23 = *(_BYTE **)(a1 + 32);
  if ( *(_BYTE **)(a1 + 24) == v23 )
    return sub_CB6200(a1, (unsigned __int8 *)"}", 1u);
  *v23 = 125;
  ++*(_QWORD *)(a1 + 32);
  return a1;
}
