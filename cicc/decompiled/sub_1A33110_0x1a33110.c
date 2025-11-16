// Function: sub_1A33110
// Address: 0x1a33110
//
__int64 __fastcall sub_1A33110(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // r13
  _QWORD *v7; // r8
  __int64 v8; // r12
  char v9; // dl
  __int64 v10; // rcx
  int v11; // esi
  unsigned int v12; // eax
  __int64 v13; // rdi
  unsigned int v15; // esi
  unsigned int v16; // eax
  _QWORD *v17; // r9
  int v18; // ecx
  unsigned int v19; // edi
  __int64 v20; // rax
  int v21; // r10d
  __int64 v22; // [rsp+0h] [rbp-30h] BYREF
  __int64 v23[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_1A246E0((__int64 *)a1, a1 + 192, **(_QWORD **)(a1 + 168));
  v5 = *(_QWORD *)(a1 + 168);
  v6 = v4;
  if ( v5 == *(_QWORD *)(a2 - 48) )
  {
    sub_1593B40((_QWORD *)(a2 - 48), v4);
    v5 = *(_QWORD *)(a1 + 168);
  }
  if ( *(_QWORD *)(a2 - 24) == v5 )
  {
    sub_1593B40((_QWORD *)(a2 - 24), v6);
    v5 = *(_QWORD *)(a1 + 168);
  }
  v23[0] = v5;
  if ( (unsigned __int8)sub_1AE9990(v5, 0) )
    sub_1A2EDE0(*(_QWORD *)(a1 + 32) + 208LL, v23);
  sub_1A22950((__int64 *)a1, a2);
  v8 = *(_QWORD *)(a1 + 184);
  v22 = a2;
  v9 = *(_BYTE *)(v8 + 8) & 1;
  if ( v9 )
  {
    v10 = v8 + 16;
    v11 = 7;
  }
  else
  {
    v15 = *(_DWORD *)(v8 + 24);
    v10 = *(_QWORD *)(v8 + 16);
    if ( !v15 )
    {
      v16 = *(_DWORD *)(v8 + 8);
      ++*(_QWORD *)v8;
      v17 = 0;
      v18 = (v16 >> 1) + 1;
LABEL_14:
      v19 = 3 * v15;
      goto LABEL_15;
    }
    v11 = v15 - 1;
  }
  v12 = v11 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (_QWORD *)(v10 + 8LL * v12);
  v13 = *v7;
  if ( a2 == *v7 )
    return 1;
  v21 = 1;
  v17 = 0;
  while ( v13 != -8 )
  {
    if ( v17 || v13 != -16 )
      v7 = v17;
    v12 = v11 & (v21 + v12);
    v13 = *(_QWORD *)(v10 + 8LL * v12);
    if ( a2 == v13 )
      return 1;
    ++v21;
    v17 = v7;
    v7 = (_QWORD *)(v10 + 8LL * v12);
  }
  v16 = *(_DWORD *)(v8 + 8);
  if ( !v17 )
    v17 = v7;
  ++*(_QWORD *)v8;
  v18 = (v16 >> 1) + 1;
  if ( !v9 )
  {
    v15 = *(_DWORD *)(v8 + 24);
    goto LABEL_14;
  }
  v19 = 24;
  v15 = 8;
LABEL_15:
  if ( v19 <= 4 * v18 )
  {
    v15 *= 2;
    goto LABEL_29;
  }
  if ( v15 - *(_DWORD *)(v8 + 12) - v18 <= v15 >> 3 )
  {
LABEL_29:
    sub_1A32DA0(v8, v15);
    sub_1A27680(v8, &v22, v23);
    v17 = (_QWORD *)v23[0];
    v16 = *(_DWORD *)(v8 + 8);
  }
  *(_DWORD *)(v8 + 8) = (2 * (v16 >> 1) + 2) | v16 & 1;
  if ( *v17 != -8 )
    --*(_DWORD *)(v8 + 12);
  *v17 = v22;
  v20 = *(unsigned int *)(v8 + 88);
  if ( (unsigned int)v20 >= *(_DWORD *)(v8 + 92) )
  {
    sub_16CD150(v8 + 80, (const void *)(v8 + 96), 0, 8, (int)v7, (int)v17);
    v20 = *(unsigned int *)(v8 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(v8 + 80) + 8 * v20) = v22;
  ++*(_DWORD *)(v8 + 88);
  return 1;
}
