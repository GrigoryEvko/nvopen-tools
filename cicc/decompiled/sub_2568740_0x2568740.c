// Function: sub_2568740
// Address: 0x2568740
//
__int64 __fastcall sub_2568740(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v4; // esi
  __int64 v5; // rcx
  int v6; // r10d
  __int64 *v7; // rbx
  __int64 v8; // rdi
  unsigned int v9; // edx
  _QWORD *v10; // rax
  __int64 v11; // r9
  unsigned __int64 *v12; // rbx
  __int64 result; // rax
  int v14; // eax
  int v15; // edx
  unsigned __int64 v16; // r12
  __int64 v17; // [rsp+0h] [rbp-40h]
  __int64 v18; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v19; // [rsp+18h] [rbp-28h] BYREF

  v2 = a1 + 168;
  v18 = a2;
  v4 = *(_DWORD *)(a1 + 192);
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 168);
    v19 = 0;
LABEL_23:
    v4 *= 2;
    goto LABEL_24;
  }
  v5 = v18;
  v6 = 1;
  v7 = 0;
  v8 = *(_QWORD *)(a1 + 176);
  v9 = (v4 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
  v10 = (_QWORD *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( v18 == *v10 )
  {
LABEL_3:
    v12 = v10 + 1;
    result = v10[1];
    if ( result )
      return result;
    goto LABEL_18;
  }
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v7 )
      v7 = v10;
    v9 = (v4 - 1) & (v6 + v9);
    v10 = (_QWORD *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( v18 == *v10 )
      goto LABEL_3;
    ++v6;
  }
  if ( !v7 )
    v7 = v10;
  v14 = *(_DWORD *)(a1 + 184);
  ++*(_QWORD *)(a1 + 168);
  v15 = v14 + 1;
  v19 = v7;
  if ( 4 * (v14 + 1) >= 3 * v4 )
    goto LABEL_23;
  if ( v4 - *(_DWORD *)(a1 + 188) - v15 <= v4 >> 3 )
  {
LABEL_24:
    sub_2512960(v2, v4);
    sub_2510430(v2, &v18, &v19);
    v5 = v18;
    v7 = v19;
    v15 = *(_DWORD *)(a1 + 184) + 1;
  }
  *(_DWORD *)(a1 + 184) = v15;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 188);
  *v7 = v5;
  v12 = (unsigned __int64 *)(v7 + 1);
  *v12 = 0;
LABEL_18:
  result = sub_22077B0(0x40u);
  if ( result )
  {
    v17 = result;
    sub_3106C40(result, a1, v18);
    result = v17;
  }
  v16 = *v12;
  *v12 = result;
  if ( v16 )
  {
    sub_C7D6A0(*(_QWORD *)(v16 + 8), 8LL * *(unsigned int *)(v16 + 24), 8);
    j_j___libc_free_0(v16);
    return *v12;
  }
  return result;
}
