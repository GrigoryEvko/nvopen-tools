// Function: sub_2573E40
// Address: 0x2573e40
//
__int64 __fastcall sub_2573E40(__int64 a1, __int64 *a2)
{
  int v4; // eax
  _QWORD *v5; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  _QWORD *v10; // r8
  __int64 v11; // r9
  __int64 result; // rax
  unsigned int v13; // esi
  __int64 v14; // r9
  __int64 *v15; // r11
  int v16; // r13d
  unsigned int v17; // edx
  __int64 *v18; // r8
  __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 *v23; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 16);
  if ( !v4 )
  {
    v5 = *(_QWORD **)(a1 + 32);
    v7 = (__int64)&v5[*(unsigned int *)(a1 + 40)];
    v10 = sub_2537FC0(v5, v7, a2);
    result = 0;
    if ( (_QWORD *)v7 == v10 )
      return sub_2573C90(a1, *a2, v8, v9, (__int64)v10, v11);
    return result;
  }
  v13 = *(_DWORD *)(a1 + 24);
  if ( !v13 )
  {
    ++*(_QWORD *)a1;
    v23 = 0;
LABEL_22:
    v13 *= 2;
    goto LABEL_23;
  }
  v14 = *(_QWORD *)(a1 + 8);
  v15 = 0;
  v16 = 1;
  v17 = (v13 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v18 = (__int64 *)(v14 + 8LL * v17);
  v19 = *v18;
  if ( *a2 == *v18 )
    return 0;
  while ( v19 != -4096 )
  {
    if ( v15 || v19 != -8192 )
      v18 = v15;
    v17 = (v13 - 1) & (v16 + v17);
    v19 = *(_QWORD *)(v14 + 8LL * v17);
    if ( *a2 == v19 )
      return 0;
    ++v16;
    v15 = v18;
    v18 = (__int64 *)(v14 + 8LL * v17);
  }
  if ( !v15 )
    v15 = v18;
  v20 = v4 + 1;
  ++*(_QWORD *)a1;
  v23 = v15;
  if ( 4 * v20 >= 3 * v13 )
    goto LABEL_22;
  if ( v13 - *(_DWORD *)(a1 + 20) - v20 <= v13 >> 3 )
  {
LABEL_23:
    sub_CF4090(a1, v13);
    sub_23FDF60(a1, a2, &v23);
    v15 = v23;
    v20 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v20;
  if ( *v15 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v15 = *a2;
  v21 = *(unsigned int *)(a1 + 40);
  v22 = *a2;
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v21 + 1, 8u, (__int64)v18, v14);
    v21 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v21) = v22;
  ++*(_DWORD *)(a1 + 40);
  return 1;
}
