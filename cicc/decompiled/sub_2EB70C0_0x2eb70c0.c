// Function: sub_2EB70C0
// Address: 0x2eb70c0
//
__int64 __fastcall sub_2EB70C0(__int64 a1)
{
  __int64 *v2; // rdx
  __int64 v3; // r14
  __int64 v4; // r15
  char v5; // al
  char v6; // r12
  __int64 v7; // r13
  __int64 v8; // rcx
  bool v9; // zf
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v13; // rdi
  int v14; // ecx
  unsigned int v15; // eax
  __int64 *v16; // r12
  __int64 v17; // rsi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned int v20; // eax
  __int64 v21; // r8
  int v22; // edi
  unsigned int v23; // eax
  _QWORD *v24; // rcx
  __int64 v25; // r9
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  unsigned int v28; // eax
  int v29; // edi
  int v30; // ecx
  int v31; // r9d
  int v32; // ecx
  int v33; // r10d
  unsigned __int64 v34; // [rsp+10h] [rbp-50h]
  _QWORD *v35; // [rsp+10h] [rbp-50h]
  _QWORD *v36; // [rsp+10h] [rbp-50h]
  char v37; // [rsp+1Fh] [rbp-41h]
  __int64 v38[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = (__int64 *)(*(_QWORD *)(a1 + 616) + 16LL * *(unsigned int *)(a1 + 624) - 16);
  v3 = v2[1];
  v4 = *v2;
  --*(_DWORD *)(a1 + 624);
  v5 = *(_BYTE *)(a1 + 608);
  v38[0] = v4;
  v37 = v5 ^ 1;
  v6 = !((v3 >> 2) & 1);
  v7 = 32LL * (v6 == (char)(v5 ^ 1));
  v8 = sub_2EB6FA0(a1, v38);
  v9 = (*(_DWORD *)(v8 + v7 + 8))-- == 1;
  if ( v9 && !*(_DWORD *)(v8 + 32LL * (unsigned __int8)(v6 ^ v37) + 8) )
  {
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v21 = a1 + 16;
      v22 = 3;
    }
    else
    {
      v29 = *(_DWORD *)(a1 + 24);
      v21 = *(_QWORD *)(a1 + 16);
      if ( !v29 )
        goto LABEL_3;
      v22 = v29 - 1;
    }
    v23 = v22 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v24 = (_QWORD *)(v21 + 72LL * v23);
    v25 = *v24;
    if ( *v24 == v4 )
    {
LABEL_17:
      v26 = v24[5];
      if ( (_QWORD *)v26 != v24 + 7 )
      {
        v35 = v24;
        _libc_free(v26);
        v24 = v35;
      }
      v27 = v24[1];
      if ( (_QWORD *)v27 != v24 + 3 )
      {
        v36 = v24;
        _libc_free(v27);
        v24 = v36;
      }
      *v24 = -8192;
      v28 = *(_DWORD *)(a1 + 8);
      ++*(_DWORD *)(a1 + 12);
      *(_DWORD *)(a1 + 8) = (2 * (v28 >> 1) - 2) | v28 & 1;
    }
    else
    {
      v32 = 1;
      while ( v25 != -4096 )
      {
        v33 = v32 + 1;
        v23 = v22 & (v32 + v23);
        v24 = (_QWORD *)(v21 + 72LL * v23);
        v25 = *v24;
        if ( *v24 == v4 )
          goto LABEL_17;
        v32 = v33;
      }
    }
  }
LABEL_3:
  v38[0] = v3 & 0xFFFFFFFFFFFFFFF8LL;
  v34 = v3 & 0xFFFFFFFFFFFFFFF8LL;
  v10 = sub_2EB6FA0(a1 + 304, v38);
  v11 = v10 + v7;
  v9 = (*(_DWORD *)(v11 + 8))-- == 1;
  if ( v9 && !*(_DWORD *)(v10 + 32LL * (unsigned __int8)(v37 ^ v6) + 8) )
  {
    if ( (*(_BYTE *)(a1 + 312) & 1) != 0 )
    {
      v13 = a1 + 320;
      v14 = 3;
    }
    else
    {
      v30 = *(_DWORD *)(a1 + 328);
      v13 = *(_QWORD *)(a1 + 320);
      if ( !v30 )
        return v4;
      v14 = v30 - 1;
    }
    v15 = v14 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
    v16 = (__int64 *)(v13 + 72LL * v15);
    v17 = *v16;
    if ( v34 == *v16 )
    {
LABEL_9:
      v18 = v16[5];
      if ( (__int64 *)v18 != v16 + 7 )
        _libc_free(v18);
      v19 = v16[1];
      if ( (__int64 *)v19 != v16 + 3 )
        _libc_free(v19);
      *v16 = -8192;
      v20 = *(_DWORD *)(a1 + 312);
      ++*(_DWORD *)(a1 + 316);
      *(_DWORD *)(a1 + 312) = (2 * (v20 >> 1) - 2) | v20 & 1;
    }
    else
    {
      v31 = 1;
      while ( v17 != -4096 )
      {
        v15 = v14 & (v31 + v15);
        v16 = (__int64 *)(v13 + 72LL * v15);
        v17 = *v16;
        if ( v34 == *v16 )
          goto LABEL_9;
        ++v31;
      }
    }
  }
  return v4;
}
