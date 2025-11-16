// Function: sub_A58D70
// Address: 0xa58d70
//
__int64 *__fastcall sub_A58D70(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  int v4; // r12d
  unsigned int v5; // esi
  __int64 v6; // r8
  __int64 v7; // rdi
  __int64 *v8; // r10
  int v9; // r14d
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rdx
  int v14; // eax
  int v15; // edx
  __int64 v16; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v17[5]; // [rsp+18h] [rbp-28h] BYREF

  v2 = a1 + 144;
  v4 = *(_DWORD *)(a1 + 176);
  v16 = a2;
  v5 = *(_DWORD *)(a1 + 168);
  *(_DWORD *)(a1 + 176) = v4 + 1;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 144);
    v17[0] = 0;
LABEL_18:
    v5 *= 2;
    goto LABEL_19;
  }
  v6 = *(_QWORD *)(a1 + 152);
  v7 = v16;
  v8 = 0;
  v9 = 1;
  v10 = (v5 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
  v11 = (__int64 *)(v6 + 16LL * v10);
  v12 = *v11;
  if ( v16 == *v11 )
  {
LABEL_3:
    *((_DWORD *)v11 + 2) = v4;
    return v11 + 1;
  }
  while ( v12 != -4096 )
  {
    if ( v12 == -8192 && !v8 )
      v8 = v11;
    v10 = (v5 - 1) & (v9 + v10);
    v11 = (__int64 *)(v6 + 16LL * v10);
    v12 = *v11;
    if ( v16 == *v11 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v8 )
    v8 = v11;
  v14 = *(_DWORD *)(a1 + 160);
  ++*(_QWORD *)(a1 + 144);
  v15 = v14 + 1;
  v17[0] = v8;
  if ( 4 * (v14 + 1) >= 3 * v5 )
    goto LABEL_18;
  if ( v5 - *(_DWORD *)(a1 + 164) - v15 <= v5 >> 3 )
  {
LABEL_19:
    sub_A429D0(v2, v5);
    sub_A56BF0(v2, &v16, v17);
    v7 = v16;
    v8 = (__int64 *)v17[0];
    v15 = *(_DWORD *)(a1 + 160) + 1;
  }
  *(_DWORD *)(a1 + 160) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 164);
  *v8 = v7;
  *((_DWORD *)v8 + 2) = 0;
  *((_DWORD *)v8 + 2) = v4;
  return v8 + 1;
}
