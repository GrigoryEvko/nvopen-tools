// Function: sub_1A9D7E0
// Address: 0x1a9d7e0
//
__int64 __fastcall sub_1A9D7E0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdx
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r10d
  __int64 v8; // r13
  unsigned int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rax
  int v14; // eax
  int v15; // ecx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // [rsp+0h] [rbp-70h] BYREF
  int v19; // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+10h] [rbp-60h] BYREF
  __int64 v21; // [rsp+18h] [rbp-58h]
  __int64 v22; // [rsp+20h] [rbp-50h]
  __int64 v23; // [rsp+28h] [rbp-48h]
  __int64 v24; // [rsp+30h] [rbp-40h]
  __int64 v25; // [rsp+38h] [rbp-38h]
  __int64 v26; // [rsp+40h] [rbp-30h]
  __int64 v27; // [rsp+48h] [rbp-28h]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  v19 = 0;
  v18 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
LABEL_23:
    v5 *= 2;
    goto LABEL_24;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = v6 + 16LL * v9;
  v11 = *(_QWORD *)v10;
  if ( v4 == *(_QWORD *)v10 )
  {
LABEL_3:
    v12 = *(unsigned int *)(v10 + 8);
    return *(_QWORD *)(a1 + 32) + (v12 << 6) + 8;
  }
  while ( v11 != -8 )
  {
    if ( !v8 && v11 == -16 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = v6 + 16LL * v9;
    v11 = *(_QWORD *)v10;
    if ( v4 == *(_QWORD *)v10 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
    goto LABEL_23;
  if ( v5 - *(_DWORD *)(a1 + 20) - v15 <= v5 >> 3 )
  {
LABEL_24:
    sub_13FEAC0(a1, v5);
    sub_13FDDE0(a1, &v18, &v20);
    v8 = v20;
    v4 = v18;
    v15 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v15;
  if ( *(_QWORD *)v8 != -8 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v8 = v4;
  *(_DWORD *)(v8 + 8) = v19;
  v16 = *a2;
  v21 = 1;
  v17 = *(_QWORD *)(a1 + 40);
  v20 = v16;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  if ( v17 == *(_QWORD *)(a1 + 48) )
  {
    sub_1A974C0((__int64 *)(a1 + 32), (char *)v17, &v20);
    if ( v25 )
      j_j___libc_free_0(v25, v27 - v25);
  }
  else
  {
    if ( v17 )
    {
      *(_QWORD *)(v17 + 24) = 0;
      *(_QWORD *)(v17 + 16) = 0;
      *(_DWORD *)(v17 + 32) = 0;
      *(_QWORD *)v17 = v16;
      *(_QWORD *)(v17 + 8) = 1;
      ++v21;
      *(_QWORD *)(v17 + 16) = v22;
      *(_QWORD *)(v17 + 24) = v23;
      *(_DWORD *)(v17 + 32) = v24;
      v22 = 0;
      v23 = 0;
      LODWORD(v24) = 0;
      *(_QWORD *)(v17 + 40) = v25;
      *(_QWORD *)(v17 + 48) = v26;
      *(_QWORD *)(v17 + 56) = v27;
      v17 = *(_QWORD *)(a1 + 40);
    }
    *(_QWORD *)(a1 + 40) = v17 + 64;
  }
  j___libc_free_0(v22);
  j___libc_free_0(0);
  v12 = (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 6) - 1;
  *(_DWORD *)(v8 + 8) = v12;
  return *(_QWORD *)(a1 + 32) + (v12 << 6) + 8;
}
