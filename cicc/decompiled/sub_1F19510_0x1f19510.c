// Function: sub_1F19510
// Address: 0x1f19510
//
void __fastcall sub_1F19510(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  unsigned int v5; // esi
  __int64 v6; // r11
  __int64 v7; // r9
  __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // rcx
  __int64 *v11; // r10
  unsigned int v12; // eax
  int v13; // ebx
  __int64 v14; // rcx
  __int64 v15; // rsi
  unsigned int v16; // eax
  int v17; // r9d
  __int64 v18; // r13
  int i; // r10d
  __int64 v20; // rdi
  __int64 v21; // r8
  _QWORD *v22; // rsi
  _QWORD *v23; // rcx
  int v24; // r9d
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax

  v4 = *(_QWORD *)a1;
  v5 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3;
  v6 = *(unsigned int *)(a1 + 16);
  v7 = *(_QWORD *)(a1 + 8) + 16 * v6 - 16;
  v8 = *(unsigned int *)(v7 + 12);
  v9 = *(_QWORD *)v7;
  LODWORD(v10) = *(_DWORD *)(v7 + 12);
  v11 = (__int64 *)(*(_QWORD *)v7 + 16 * v8 + 8);
  v12 = *(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v11 >> 1) & 3;
  if ( !*(_DWORD *)(v4 + 184) )
  {
    if ( v5 < v12 )
      goto LABEL_5;
    v16 = v8 + 1;
    if ( *(_DWORD *)(v7 + 8) <= (unsigned int)(v8 + 1)
      || *(_DWORD *)(v9 + 4 * v8 + 144) != *(_DWORD *)(v9 + 4LL * v16 + 144)
      || *(_QWORD *)(v9 + 16LL * v16) != a2 )
    {
      goto LABEL_5;
    }
    goto LABEL_11;
  }
  if ( v5 < v12 )
    goto LABEL_5;
  v13 = *(_DWORD *)(v9 + 4 * v8 + 144);
  v14 = (unsigned int)(v10 + 1);
  if ( *(_DWORD *)(v7 + 8) <= (unsigned int)v14 )
  {
    v26 = sub_3945FF0(a1 + 8, (unsigned int)(v6 - 1));
    if ( !v26
      || (v4 = *(_QWORD *)a1, v27 = v26 & 0xFFFFFFFFFFFFFFC0LL, *(_DWORD *)(v27 + 144) != v13)
      || *(_QWORD *)v27 != a2 )
    {
      v28 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
      v11 = (__int64 *)(*(_QWORD *)v28 + 16LL * *(unsigned int *)(v28 + 12) + 8);
      goto LABEL_5;
    }
    v29 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
    v10 = *(unsigned int *)(v29 + 12);
    v9 = *(_QWORD *)v29;
    v8 = v10;
    if ( *(_DWORD *)(*(_QWORD *)a1 + 184LL) )
    {
LABEL_16:
      v18 = *(_QWORD *)(v9 + 16 * v8);
      sub_1F192F0(a1, 1, v4, v10, v9);
LABEL_14:
      v25 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
      *(_QWORD *)(*(_QWORD *)v25 + 16LL * *(unsigned int *)(v25 + 12)) = v18;
      return;
    }
    v16 = v10 + 1;
    v8 = (unsigned int)v10;
LABEL_11:
    v17 = *(_DWORD *)(v4 + 188);
    v18 = *(_QWORD *)(v9 + 16 * v8);
    for ( i = v10 - v16; v17 != v16; *(_DWORD *)(v4 + 4 * v21 + 144) = *(_DWORD *)(v4 + 4 * v20 + 144) )
    {
      v20 = v16;
      v21 = i + v16++;
      v22 = (_QWORD *)(v4 + 16 * v20);
      v23 = (_QWORD *)(v4 + 16 * v21);
      *v23 = *v22;
      v23[1] = v22[1];
    }
    v24 = v17 - 1;
    *(_DWORD *)(v4 + 188) = v24;
    *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v24;
    goto LABEL_14;
  }
  if ( *(_DWORD *)(v9 + 4 * v14 + 144) == v13 )
  {
    v10 = 16 * v14;
    if ( *(_QWORD *)(v9 + v10) == a2 )
      goto LABEL_16;
  }
LABEL_5:
  *v11 = a2;
  v15 = (unsigned int)(*(_DWORD *)(a1 + 16) - 1);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16 * v15 + 12) == *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16 * v15 + 8) - 1 )
    sub_1F18EF0(a1, v15, a2);
}
