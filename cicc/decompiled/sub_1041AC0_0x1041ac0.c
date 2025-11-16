// Function: sub_1041AC0
// Address: 0x1041ac0
//
__int64 __fastcall sub_1041AC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v3; // r8d
  __int64 v4; // rcx
  int v5; // r10d
  __int64 *v6; // r12
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 result; // rax
  int v11; // eax
  int v12; // edx
  __int64 v13; // rdi
  int v14; // esi
  __int64 *v15; // [rsp+8h] [rbp-38h] BYREF
  __int64 v16; // [rsp+10h] [rbp-30h] BYREF
  __int64 v17; // [rsp+18h] [rbp-28h]

  v2 = a1 + 96;
  v3 = *(_DWORD *)(a1 + 120);
  v16 = a2;
  v17 = 0;
  if ( !v3 )
  {
    ++*(_QWORD *)(a1 + 96);
    v15 = 0;
LABEL_22:
    v14 = 2 * v3;
LABEL_23:
    sub_10418C0(v2, v14);
    sub_103F7C0(v2, &v16, &v15);
    a2 = v16;
    v6 = v15;
    v12 = *(_DWORD *)(a1 + 112) + 1;
    goto LABEL_15;
  }
  v4 = *(_QWORD *)(a1 + 104);
  v5 = 1;
  v6 = 0;
  v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v4 + 16LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
    return v8[1];
  while ( v9 != -4096 )
  {
    if ( v9 == -8192 && !v6 )
      v6 = v8;
    v7 = (v3 - 1) & (v5 + v7);
    v8 = (__int64 *)(v4 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
      return v8[1];
    ++v5;
  }
  if ( !v6 )
    v6 = v8;
  v11 = *(_DWORD *)(a1 + 112);
  ++*(_QWORD *)(a1 + 96);
  v12 = v11 + 1;
  v15 = v6;
  if ( 4 * (v11 + 1) >= 3 * v3 )
    goto LABEL_22;
  if ( v3 - *(_DWORD *)(a1 + 116) - v12 <= v3 >> 3 )
  {
    v14 = v3;
    goto LABEL_23;
  }
LABEL_15:
  *(_DWORD *)(a1 + 112) = v12;
  if ( *v6 != -4096 )
    --*(_DWORD *)(a1 + 116);
  *v6 = a2;
  v6[1] = v17;
  result = sub_22077B0(16);
  if ( result )
  {
    *(_QWORD *)(result + 8) = result;
    *(_QWORD *)result = result | 4;
  }
  v13 = v6[1];
  v6[1] = result;
  if ( v13 )
  {
    j_j___libc_free_0(v13, 16);
    return v6[1];
  }
  return result;
}
