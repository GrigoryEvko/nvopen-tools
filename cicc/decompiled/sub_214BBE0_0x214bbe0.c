// Function: sub_214BBE0
// Address: 0x214bbe0
//
__int64 __fastcall sub_214BBE0(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // r8
  unsigned __int64 v5; // r8
  __int64 v6; // rdi
  __int64 v7; // rax
  unsigned int v8; // ecx
  __int64 *v9; // rsi
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdi
  unsigned int v13; // esi
  int *v14; // rcx
  int v15; // r9d
  unsigned int v16; // ebx
  __int64 v17; // rax
  int v19; // ecx
  int v20; // esi
  int v21; // r11d
  int v22; // r11d
  char *v23[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v24; // [rsp+10h] [rbp-60h] BYREF
  void *v25; // [rsp+20h] [rbp-50h] BYREF
  __int64 v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+30h] [rbp-40h]
  __int64 v28; // [rsp+38h] [rbp-38h]
  int v29; // [rsp+40h] [rbp-30h]
  __int64 v30; // [rsp+48h] [rbp-28h]

  v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 800) + 24LL) + 16LL * (a3 & 0x7FFFFFFF));
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = a1 + 16;
  *(_BYTE *)(a1 + 16) = 0;
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v30 = a1;
  v6 = *(_QWORD *)(a2 + 816);
  v25 = &unk_49EFBE0;
  v7 = *(unsigned int *)(a2 + 832);
  v29 = 1;
  v28 = 0;
  v27 = 0;
  v26 = 0;
  if ( (_DWORD)v7 )
  {
    v8 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v9 = (__int64 *)(v6 + 40LL * v8);
    v10 = *v9;
    if ( v5 == *v9 )
      goto LABEL_3;
    v20 = 1;
    while ( v10 != -8 )
    {
      v22 = v20 + 1;
      v8 = (v7 - 1) & (v20 + v8);
      v9 = (__int64 *)(v6 + 40LL * v8);
      v10 = *v9;
      if ( v5 == *v9 )
        goto LABEL_3;
      v20 = v22;
    }
  }
  v9 = (__int64 *)(v6 + 40 * v7);
LABEL_3:
  v11 = *((unsigned int *)v9 + 8);
  v12 = v9[2];
  if ( (_DWORD)v11 )
  {
    v13 = (v11 - 1) & (37 * a3);
    v14 = (int *)(v12 + 8LL * v13);
    v15 = *v14;
    if ( *v14 == a3 )
      goto LABEL_5;
    v19 = 1;
    while ( v15 != -1 )
    {
      v21 = v19 + 1;
      v13 = (v11 - 1) & (v19 + v13);
      v14 = (int *)(v12 + 8LL * v13);
      v15 = *v14;
      if ( *v14 == a3 )
        goto LABEL_5;
      v19 = v21;
    }
  }
  v14 = (int *)(v12 + 8 * v11);
LABEL_5:
  v16 = v14[1];
  sub_21638D0(v23, v5);
  v17 = sub_16E7EE0((__int64)&v25, v23[0], (size_t)v23[1]);
  sub_16E7A90(v17, v16);
  if ( (__int64 *)v23[0] != &v24 )
    j_j___libc_free_0(v23[0], v24 + 1);
  if ( v26 != v28 )
    sub_16E7BA0((__int64 *)&v25);
  sub_16E7BC0((__int64 *)&v25);
  return a1;
}
