// Function: sub_A58EF0
// Address: 0xa58ef0
//
__int64 *__fastcall sub_A58EF0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v4; // r12d
  unsigned int v5; // r8d
  __int64 v6; // rdi
  int v7; // r14d
  __int64 *v8; // r10
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // rdx
  int v13; // eax
  int v14; // edx
  int v15; // esi
  __int64 v16; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v17[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 104;
  v4 = *(_DWORD *)(a1 + 136);
  v5 = *(_DWORD *)(a1 + 128);
  v16 = a2;
  *(_DWORD *)(a1 + 136) = v4 + 1;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 104);
    v17[0] = 0;
LABEL_18:
    v15 = 2 * v5;
LABEL_19:
    sub_A429D0(v2, v15);
    sub_A56BF0(v2, &v16, v17);
    a2 = v16;
    v8 = (__int64 *)v17[0];
    v14 = *(_DWORD *)(a1 + 120) + 1;
    goto LABEL_14;
  }
  v6 = *(_QWORD *)(a1 + 112);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
  {
LABEL_3:
    *((_DWORD *)v10 + 2) = v4;
    return v10 + 1;
  }
  while ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v13 = *(_DWORD *)(a1 + 120);
  ++*(_QWORD *)(a1 + 104);
  v14 = v13 + 1;
  v17[0] = v8;
  if ( 4 * (v13 + 1) >= 3 * v5 )
    goto LABEL_18;
  if ( v5 - *(_DWORD *)(a1 + 124) - v14 <= v5 >> 3 )
  {
    v15 = v5;
    goto LABEL_19;
  }
LABEL_14:
  *(_DWORD *)(a1 + 120) = v14;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 124);
  *v8 = a2;
  *((_DWORD *)v8 + 2) = 0;
  *((_DWORD *)v8 + 2) = v4;
  return v8 + 1;
}
