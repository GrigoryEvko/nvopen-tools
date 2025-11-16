// Function: sub_1081170
// Address: 0x1081170
//
void __fastcall sub_1081170(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned int v7; // edi
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r9
  int v12; // r14d
  unsigned int v13; // esi
  int v14; // r13d
  __int64 v15; // r9
  int v16; // r11d
  __int64 v17; // r8
  unsigned int v18; // ecx
  __int64 *v19; // rdx
  __int64 v20; // rdi
  int v21; // edx
  int v22; // edi
  int v23; // ecx
  __int64 v24; // rbx
  __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rsi
  int v29; // r10d
  __int64 v30; // [rsp-48h] [rbp-48h] BYREF
  __int64 v31; // [rsp-40h] [rbp-40h] BYREF

  if ( (unsigned int)a3 > 0x18 )
    return;
  v3 = 17567750;
  if ( !_bittest64(&v3, a3) )
    return;
  v5 = sub_E5C930(*(__int64 **)a1, a2);
  v6 = *(_QWORD *)(a1 + 8);
  v30 = v5;
  v7 = *(_DWORD *)(v6 + 256);
  v8 = *(_QWORD *)(v6 + 240);
  if ( v7 )
  {
    v9 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( v5 == *v10 )
      goto LABEL_5;
    v21 = 1;
    while ( v11 != -4096 )
    {
      v29 = v21 + 1;
      v9 = (v7 - 1) & (v21 + v9);
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v5 == *v10 )
        goto LABEL_5;
      v21 = v29;
    }
  }
  v10 = (__int64 *)(v8 + 16LL * v7);
LABEL_5:
  v12 = *((_DWORD *)v10 + 2);
  v13 = *(_DWORD *)(v6 + 224);
  v14 = *(_DWORD *)(*(_QWORD *)(a1 + 16) + 8LL);
  if ( !v13 )
  {
    ++*(_QWORD *)(v6 + 200);
    v31 = 0;
LABEL_26:
    v13 *= 2;
    goto LABEL_27;
  }
  v15 = *(_QWORD *)(v6 + 208);
  v16 = 1;
  v17 = 0;
  v18 = (v13 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v19 = (__int64 *)(v15 + 16LL * v18);
  v20 = *v19;
  if ( v5 == *v19 )
    return;
  while ( v20 != -4096 )
  {
    if ( v17 || v20 != -8192 )
      v19 = (__int64 *)v17;
    v18 = (v13 - 1) & (v16 + v18);
    v20 = *(_QWORD *)(v15 + 16LL * v18);
    if ( v5 == v20 )
      return;
    ++v16;
    v17 = (__int64)v19;
    v19 = (__int64 *)(v15 + 16LL * v18);
  }
  v22 = *(_DWORD *)(v6 + 216);
  if ( !v17 )
    v17 = (__int64)v19;
  ++*(_QWORD *)(v6 + 200);
  v23 = v22 + 1;
  v31 = v17;
  if ( 4 * (v22 + 1) >= 3 * v13 )
    goto LABEL_26;
  if ( v13 - *(_DWORD *)(v6 + 220) - v23 <= v13 >> 3 )
  {
LABEL_27:
    sub_107DA80(v6 + 200, v13);
    sub_107C950(v6 + 200, &v30, &v31);
    v5 = v30;
    v17 = v31;
    v23 = *(_DWORD *)(v6 + 216) + 1;
  }
  *(_DWORD *)(v6 + 216) = v23;
  if ( *(_QWORD *)v17 != -4096 )
    --*(_DWORD *)(v6 + 220);
  *(_QWORD *)v17 = v5;
  *(_DWORD *)(v17 + 8) = v14 + 1;
  v24 = *(_QWORD *)(a1 + 16);
  v25 = *(unsigned int *)(v24 + 8);
  v26 = *(unsigned int *)(v24 + 12);
  if ( v25 + 1 > v26 )
  {
    sub_C8D5F0(*(_QWORD *)(a1 + 16), (const void *)(v24 + 16), v25 + 1, 4u, v17, v15);
    v25 = *(unsigned int *)(v24 + 8);
  }
  v27 = *(_QWORD *)v24;
  *(_DWORD *)(v27 + 4 * v25) = v12;
  v28 = v30;
  ++*(_DWORD *)(v24 + 8);
  sub_107FDF0(*(_QWORD *)(a1 + 8), v28, v27, v26, v17, v15);
}
