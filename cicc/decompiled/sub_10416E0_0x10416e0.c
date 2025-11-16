// Function: sub_10416E0
// Address: 0x10416e0
//
__int64 __fastcall sub_10416E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v3; // r8d
  __int64 v4; // rcx
  int v5; // r10d
  __int64 *v6; // r13
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 result; // rax
  int v11; // eax
  int v12; // edx
  unsigned __int64 *v13; // r12
  unsigned __int64 *v14; // rbx
  unsigned __int64 *v15; // rdi
  unsigned __int64 v16; // rdx
  int v17; // esi
  __int64 *v18; // [rsp+8h] [rbp-38h] BYREF
  __int64 v19; // [rsp+10h] [rbp-30h] BYREF
  __int64 v20; // [rsp+18h] [rbp-28h]

  v2 = a1 + 64;
  v3 = *(_DWORD *)(a1 + 88);
  v19 = a2;
  v20 = 0;
  if ( !v3 )
  {
    ++*(_QWORD *)(a1 + 64);
    v18 = 0;
LABEL_24:
    v17 = 2 * v3;
LABEL_25:
    sub_1041470(v2, v17);
    sub_103F700(v2, &v19, &v18);
    a2 = v19;
    v6 = v18;
    v12 = *(_DWORD *)(a1 + 80) + 1;
    goto LABEL_15;
  }
  v4 = *(_QWORD *)(a1 + 72);
  v5 = 1;
  v6 = 0;
  v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v4 + 16LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
    return v8[1];
  while ( v9 != -4096 )
  {
    if ( !v6 && v9 == -8192 )
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
  v11 = *(_DWORD *)(a1 + 80);
  ++*(_QWORD *)(a1 + 64);
  v12 = v11 + 1;
  v18 = v6;
  if ( 4 * (v11 + 1) >= 3 * v3 )
    goto LABEL_24;
  if ( v3 - *(_DWORD *)(a1 + 84) - v12 <= v3 >> 3 )
  {
    v17 = v3;
    goto LABEL_25;
  }
LABEL_15:
  *(_DWORD *)(a1 + 80) = v12;
  if ( *v6 != -4096 )
    --*(_DWORD *)(a1 + 84);
  *v6 = a2;
  v6[1] = v20;
  result = sub_22077B0(16);
  if ( result )
  {
    *(_QWORD *)(result + 8) = result;
    *(_QWORD *)result = result | 4;
  }
  v13 = (unsigned __int64 *)v6[1];
  v6[1] = result;
  if ( v13 )
  {
    v14 = (unsigned __int64 *)v13[1];
    while ( v13 != v14 )
    {
      v15 = v14;
      v14 = (unsigned __int64 *)v14[1];
      v16 = *v15 & 0xFFFFFFFFFFFFFFF8LL;
      *v14 = v16 | *v14 & 7;
      *(_QWORD *)(v16 + 8) = v14;
      *v15 &= 7u;
      v15 -= 4;
      v15[5] = 0;
      sub_BD72D0((__int64)v15, a2);
    }
    j_j___libc_free_0(v13, 16);
    return v6[1];
  }
  return result;
}
