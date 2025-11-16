// Function: sub_D50980
// Address: 0xd50980
//
void __fastcall sub_D50980(__int64 a1, __int64 a2)
{
  unsigned int v4; // esi
  __int64 v5; // rdi
  unsigned int v6; // eax
  __int64 *v7; // rcx
  __int64 v8; // rdx
  __int64 *v9; // r13
  __int64 *i; // rbx
  __int64 v11; // rdi
  int v12; // r10d
  __int64 *v13; // r9
  int v14; // eax
  int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v18; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a2 + 24);
  v17 = a1;
  if ( !v4 )
  {
    ++*(_QWORD *)a2;
    v18 = 0;
LABEL_16:
    v4 *= 2;
    goto LABEL_17;
  }
  v5 = *(_QWORD *)(a2 + 8);
  v6 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v7 = (__int64 *)(v5 + 8LL * v6);
  v8 = *v7;
  if ( a1 == *v7 )
    goto LABEL_3;
  v12 = 1;
  v13 = 0;
  while ( v8 != -4096 )
  {
    if ( v13 || v8 != -8192 )
      v7 = v13;
    v6 = (v4 - 1) & (v12 + v6);
    v8 = *(_QWORD *)(v5 + 8LL * v6);
    if ( a1 == v8 )
      goto LABEL_3;
    ++v12;
    v13 = v7;
    v7 = (__int64 *)(v5 + 8LL * v6);
  }
  v14 = *(_DWORD *)(a2 + 16);
  if ( !v13 )
    v13 = v7;
  ++*(_QWORD *)a2;
  v15 = v14 + 1;
  v18 = v13;
  if ( 4 * (v14 + 1) >= 3 * v4 )
    goto LABEL_16;
  v16 = a1;
  if ( v4 - *(_DWORD *)(a2 + 20) - v15 <= v4 >> 3 )
  {
LABEL_17:
    sub_D507B0(a2, v4);
    sub_D4D180(a2, &v17, &v18);
    v16 = v17;
    v13 = v18;
    v15 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v15;
  if ( *v13 != -4096 )
    --*(_DWORD *)(a2 + 20);
  *v13 = v16;
LABEL_3:
  nullsub_188();
  v9 = *(__int64 **)(a1 + 16);
  for ( i = *(__int64 **)(a1 + 8); v9 != i; ++i )
  {
    v11 = *i;
    sub_D50980(v11, a2);
  }
}
