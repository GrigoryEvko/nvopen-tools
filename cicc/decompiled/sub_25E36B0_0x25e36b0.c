// Function: sub_25E36B0
// Address: 0x25e36b0
//
bool __fastcall sub_25E36B0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // r8
  __int64 v7; // rdi
  int v8; // r14d
  __int64 *v9; // r11
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r10
  unsigned __int64 v13; // rbx
  int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // [rsp+0h] [rbp-40h] BYREF
  __int64 v19; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v20; // [rsp+18h] [rbp-28h] BYREF

  v4 = *a1;
  v19 = a2;
  v18 = a3;
  v5 = *(_DWORD *)(v4 + 24);
  if ( !v5 )
  {
    v20 = 0;
    ++*(_QWORD *)v4;
LABEL_19:
    v5 *= 2;
    goto LABEL_20;
  }
  v6 = *(_QWORD *)(v4 + 8);
  v7 = v4;
  v8 = 1;
  v9 = 0;
  v10 = (v5 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
  v11 = (__int64 *)(v6 + 16LL * v10);
  v12 = *v11;
  if ( v19 == *v11 )
  {
LABEL_3:
    v13 = v11[1];
    return *sub_9DDC30(v7, &v18) < v13;
  }
  while ( v12 != -4096 )
  {
    if ( v12 == -8192 && !v9 )
      v9 = v11;
    v10 = (v5 - 1) & (v8 + v10);
    v11 = (__int64 *)(v6 + 16LL * v10);
    v12 = *v11;
    if ( v19 == *v11 )
      goto LABEL_3;
    ++v8;
  }
  if ( !v9 )
    v9 = v11;
  v15 = *(_DWORD *)(v4 + 16) + 1;
  v20 = v9;
  ++*(_QWORD *)v4;
  if ( 4 * v15 >= 3 * v5 )
    goto LABEL_19;
  if ( v5 - *(_DWORD *)(v4 + 20) - v15 <= v5 >> 3 )
  {
LABEL_20:
    sub_9DDA50(v4, v5);
    sub_25E0C90(v4, &v19, &v20);
    v15 = *(_DWORD *)(v4 + 16) + 1;
  }
  v16 = v20;
  *(_DWORD *)(v4 + 16) = v15;
  if ( *v16 != -4096 )
    --*(_DWORD *)(v4 + 20);
  v17 = v19;
  v16[1] = 0;
  v13 = 0;
  *v16 = v17;
  v7 = *a1;
  return *sub_9DDC30(v7, &v18) < v13;
}
