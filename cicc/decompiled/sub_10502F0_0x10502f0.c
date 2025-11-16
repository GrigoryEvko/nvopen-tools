// Function: sub_10502F0
// Address: 0x10502f0
//
__int64 __fastcall sub_10502F0(_QWORD *a1, __int64 a2, int *a3)
{
  __int64 v6; // r13
  _QWORD *v7; // r8
  unsigned int v8; // esi
  __int64 v9; // r10
  __int64 v10; // r9
  unsigned int v11; // edx
  __int64 v12; // r15
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // rax
  int v19; // ecx
  _QWORD *v20; // rdx
  unsigned int v21; // eax
  unsigned int v22; // ecx
  __int64 v23; // rsi
  __int64 result; // rax
  __int64 *v25; // rax
  int v26; // edi
  int v27; // edx
  __int64 v28; // rdx
  __int64 *v29; // rdx
  int v30; // esi
  int v31; // esi
  __int64 v32; // r10
  unsigned int v33; // ecx
  int v34; // r15d
  __int64 *v35; // r11
  int v36; // esi
  int v37; // esi
  int v38; // r15d
  __int64 v39; // r10
  unsigned int v40; // ecx
  int v41; // [rsp+8h] [rbp-48h]
  _QWORD *v42; // [rsp+8h] [rbp-48h]
  _QWORD *v43; // [rsp+8h] [rbp-48h]
  __int64 v44; // [rsp+14h] [rbp-3Ch]
  int v45; // [rsp+1Ch] [rbp-34h]

  v6 = *a1;
  v7 = (_QWORD *)a1[1];
  v8 = *(_DWORD *)(*a1 + 1352LL);
  v9 = *a1 + 1328LL;
  if ( !v8 )
  {
    ++*(_QWORD *)(v6 + 1328);
    goto LABEL_21;
  }
  v10 = *(_QWORD *)(v6 + 1336);
  v11 = (v8 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
  v12 = v10 + 72LL * v11;
  v13 = *(_QWORD *)v12;
  if ( *v7 == *(_QWORD *)v12 )
  {
LABEL_3:
    v14 = *(unsigned int *)(v12 + 16);
    v15 = (__int64 *)(v12 + 8);
    v10 = *(unsigned int *)(v6 + 56);
    v7 = (_QWORD *)(v14 + 1);
    HIDWORD(v44) = *a3;
    LODWORD(v44) = *(_DWORD *)(v6 + 56);
    LOBYTE(v45) = *((_BYTE *)a3 + 4);
    if ( *(unsigned int *)(v12 + 20) < (unsigned __int64)(v14 + 1) )
    {
      sub_C8D5F0(v12 + 8, (const void *)(v12 + 24), v14 + 1, 0xCu, (__int64)v7, v10);
      v14 = *(unsigned int *)(v12 + 16);
      v15 = (__int64 *)(v12 + 8);
    }
    goto LABEL_5;
  }
  v41 = 1;
  v25 = 0;
  while ( v13 != -4096 )
  {
    if ( v13 == -8192 && !v25 )
      v25 = (__int64 *)v12;
    v11 = (v8 - 1) & (v41 + v11);
    v12 = v10 + 72LL * v11;
    v13 = *(_QWORD *)v12;
    if ( *v7 == *(_QWORD *)v12 )
      goto LABEL_3;
    ++v41;
  }
  v26 = *(_DWORD *)(v6 + 1344);
  if ( !v25 )
    v25 = (__int64 *)v12;
  ++*(_QWORD *)(v6 + 1328);
  v27 = v26 + 1;
  if ( 4 * (v26 + 1) >= 3 * v8 )
  {
LABEL_21:
    v42 = v7;
    sub_104FFE0(v9, 2 * v8);
    v30 = *(_DWORD *)(v6 + 1352);
    if ( v30 )
    {
      v7 = v42;
      v31 = v30 - 1;
      v32 = *(_QWORD *)(v6 + 1336);
      v27 = *(_DWORD *)(v6 + 1344) + 1;
      v33 = v31 & (((unsigned int)*v42 >> 9) ^ ((unsigned int)*v42 >> 4));
      v25 = (__int64 *)(v32 + 72LL * v33);
      v10 = *v25;
      if ( *v42 == *v25 )
        goto LABEL_17;
      v34 = 1;
      v35 = 0;
      while ( v10 != -4096 )
      {
        if ( !v35 && v10 == -8192 )
          v35 = v25;
        v33 = v31 & (v34 + v33);
        v25 = (__int64 *)(v32 + 72LL * v33);
        v10 = *v25;
        if ( *v42 == *v25 )
          goto LABEL_17;
        ++v34;
      }
LABEL_25:
      if ( v35 )
        v25 = v35;
      goto LABEL_17;
    }
LABEL_46:
    ++*(_DWORD *)(v6 + 1344);
    BUG();
  }
  if ( v8 - *(_DWORD *)(v6 + 1348) - v27 <= v8 >> 3 )
  {
    v43 = v7;
    sub_104FFE0(v9, v8);
    v36 = *(_DWORD *)(v6 + 1352);
    if ( v36 )
    {
      v7 = v43;
      v37 = v36 - 1;
      v38 = 1;
      v35 = 0;
      v39 = *(_QWORD *)(v6 + 1336);
      v27 = *(_DWORD *)(v6 + 1344) + 1;
      v40 = v37 & (((unsigned int)*v43 >> 9) ^ ((unsigned int)*v43 >> 4));
      v25 = (__int64 *)(v39 + 72LL * v40);
      v10 = *v25;
      if ( *v25 == *v43 )
        goto LABEL_17;
      while ( v10 != -4096 )
      {
        if ( v10 == -8192 && !v35 )
          v35 = v25;
        v40 = v37 & (v38 + v40);
        v25 = (__int64 *)(v39 + 72LL * v40);
        v10 = *v25;
        if ( *v43 == *v25 )
          goto LABEL_17;
        ++v38;
      }
      goto LABEL_25;
    }
    goto LABEL_46;
  }
LABEL_17:
  *(_DWORD *)(v6 + 1344) = v27;
  if ( *v25 != -4096 )
    --*(_DWORD *)(v6 + 1348);
  v28 = *v7;
  v25[2] = 0x400000000LL;
  *v25 = v28;
  v29 = v25 + 3;
  v15 = v25 + 1;
  *v15 = (__int64)v29;
  LOBYTE(v45) = *((_BYTE *)a3 + 4);
  v14 = 0;
  LODWORD(v44) = *(_DWORD *)(*a1 + 56LL);
  HIDWORD(v44) = *a3;
LABEL_5:
  v16 = *v15 + 12 * v14;
  *(_QWORD *)v16 = v44;
  *(_DWORD *)(v16 + 8) = v45;
  ++*((_DWORD *)v15 + 2);
  v17 = *a1;
  v18 = *(unsigned int *)(*a1 + 56LL);
  if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 60LL) )
  {
    sub_C8D5F0(v17 + 48, (const void *)(v17 + 64), v18 + 1, 8u, (__int64)v7, v10);
    v18 = *(unsigned int *)(v17 + 56);
  }
  *(_QWORD *)(*(_QWORD *)(v17 + 48) + 8 * v18) = a2;
  ++*(_DWORD *)(v17 + 56);
  v19 = *a3;
  v20 = (_QWORD *)a1[2];
  v21 = *a3;
  if ( *((_BYTE *)a3 + 4) )
  {
    *(_QWORD *)(v20[9] + 8LL * (v21 >> 6)) &= ~(1LL << v19);
    v22 = *a3;
    v23 = *(_QWORD *)a1[2];
  }
  else
  {
    *(_QWORD *)(*v20 + 8LL * (v21 >> 6)) &= ~(1LL << v19);
    v22 = *a3;
    v23 = *(_QWORD *)(a1[2] + 72LL);
  }
  result = v22 >> 6;
  *(_QWORD *)(v23 + 8 * result) |= 1LL << v22;
  return result;
}
