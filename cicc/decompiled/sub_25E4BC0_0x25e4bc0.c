// Function: sub_25E4BC0
// Address: 0x25e4bc0
//
bool __fastcall sub_25E4BC0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  unsigned __int64 *v5; // rax
  __int64 v6; // rbx
  unsigned __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // rdi
  __int64 *v10; // r10
  int v11; // r11d
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r9
  unsigned __int64 v15; // rax
  int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // [rsp+0h] [rbp-40h] BYREF
  __int64 v21; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v22; // [rsp+18h] [rbp-28h] BYREF

  v4 = *a1;
  v21 = a2;
  v20 = a3;
  v5 = sub_9DDC30(v4, &v21);
  v6 = *a1;
  v7 = *v5;
  v8 = *(_DWORD *)(v6 + 24);
  if ( !v8 )
  {
    v22 = 0;
    ++*(_QWORD *)v6;
LABEL_19:
    v8 *= 2;
    goto LABEL_20;
  }
  v9 = *(_QWORD *)(v6 + 8);
  v10 = 0;
  v11 = 1;
  v12 = (v8 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
  v13 = (__int64 *)(v9 + 16LL * v12);
  v14 = *v13;
  if ( v20 == *v13 )
  {
LABEL_3:
    v15 = v13[1];
    return v7 > v15;
  }
  while ( v14 != -4096 )
  {
    if ( v14 == -8192 && !v10 )
      v10 = v13;
    v12 = (v8 - 1) & (v11 + v12);
    v13 = (__int64 *)(v9 + 16LL * v12);
    v14 = *v13;
    if ( v20 == *v13 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v10 )
    v10 = v13;
  v17 = *(_DWORD *)(v6 + 16) + 1;
  v22 = v10;
  ++*(_QWORD *)v6;
  if ( 4 * v17 >= 3 * v8 )
    goto LABEL_19;
  if ( v8 - *(_DWORD *)(v6 + 20) - v17 <= v8 >> 3 )
  {
LABEL_20:
    sub_9DDA50(v6, v8);
    sub_25E0C90(v6, &v20, &v22);
    v17 = *(_DWORD *)(v6 + 16) + 1;
  }
  v18 = v22;
  *(_DWORD *)(v6 + 16) = v17;
  if ( *v18 != -4096 )
    --*(_DWORD *)(v6 + 20);
  v19 = v20;
  v18[1] = 0;
  *v18 = v19;
  v15 = 0;
  return v7 > v15;
}
