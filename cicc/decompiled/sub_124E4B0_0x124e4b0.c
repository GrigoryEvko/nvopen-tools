// Function: sub_124E4B0
// Address: 0x124e4b0
//
unsigned __int64 __fastcall sub_124E4B0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r13
  unsigned int v6; // esi
  __int64 v7; // r10
  __int64 v9; // r8
  unsigned int v10; // eax
  unsigned int v11; // r15d
  __int64 *v12; // rdx
  int v13; // r15d
  int v14; // r11d
  unsigned int v15; // edi
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // r15
  unsigned int v20; // r13d
  bool v21; // zf
  char v22; // al
  int v23; // edx
  bool v24; // cf
  char *v25; // rax
  unsigned __int64 v26; // r9
  int v27; // r8d
  unsigned __int64 result; // rax
  int v29; // eax
  int v30; // ecx
  __int64 v31; // r9
  int v32; // eax
  int v33; // esi
  __int64 v34; // r8
  unsigned int v35; // eax
  __int64 v36; // rdi
  int v37; // r10d
  __int64 *v38; // r9
  int v39; // eax
  int v40; // eax
  __int64 v41; // rdi
  __int64 *v42; // r8
  unsigned int v43; // r15d
  int v44; // r9d
  __int64 v45; // rsi
  __int64 v46; // [rsp+8h] [rbp-68h]
  void *v47[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v48; // [rsp+20h] [rbp-50h]
  __int64 v49; // [rsp+28h] [rbp-48h]
  __int16 v50; // [rsp+30h] [rbp-40h]

  v5 = *a1;
  v6 = *(_DWORD *)(*a1 + 160);
  v7 = *a1 + 136;
  if ( !v6 )
  {
    ++*(_QWORD *)(v5 + 136);
    goto LABEL_25;
  }
  v9 = *(_QWORD *)(v5 + 144);
  v10 = (unsigned int)a3 >> 9;
  v11 = (unsigned int)a3 >> 4;
  v12 = 0;
  v13 = v10 ^ v11;
  v14 = 1;
  v15 = (v6 - 1) & v13;
  v16 = (__int64 *)(v9 + 32LL * v15);
  v17 = *v16;
  if ( a3 == *v16 )
  {
LABEL_3:
    if ( v16[1] == v16[2] )
      return 0;
    v18 = *(_QWORD *)(a2 + 2368);
    v19 = *(_QWORD *)(a3 + 136);
    v20 = (*(_DWORD *)(a3 + 152) & 0x200) == 0 ? 64 : 512;
    if ( v18 && *(_BYTE *)(v18 + 2) )
    {
      v48 = *(_QWORD *)(a3 + 128);
      v31 = *(_QWORD *)(a3 + 168);
      v49 = v19;
      v50 = 1283;
      v47[0] = ".crel";
      return sub_E6CD60(a2, v47, 1073741844, v20, 1, v31 & 0xFFFFFFFFFFFFFFF8LL, a3);
    }
    else
    {
      v46 = *(_QWORD *)(a3 + 128);
      v21 = (unsigned __int8)sub_124CB30(*a1, v18, a3) == 0;
      v22 = *(_BYTE *)(*(_QWORD *)(*a1 + 112) + 12LL);
      if ( v21 )
      {
        v23 = 9;
        v24 = (v22 & 2) == 0;
        v25 = ".rel";
        v26 = *(_QWORD *)(a3 + 168) & 0xFFFFFFFFFFFFFFF8LL;
        v27 = v24 ? 8 : 16;
      }
      else
      {
        v23 = 4;
        v24 = (v22 & 2) == 0;
        v25 = ".rela";
        v26 = *(_QWORD *)(a3 + 168) & 0xFFFFFFFFFFFFFFF8LL;
        v27 = v24 ? 12 : 24;
      }
      v48 = v46;
      v50 = 1283;
      v47[0] = v25;
      v49 = v19;
      result = sub_E6CD60(a2, v47, v23, v20, v27, v26, a3);
      *(_BYTE *)(result + 32) = ((*(_BYTE *)(*(_QWORD *)(*a1 + 112) + 12LL) & 2) != 0) + 2;
    }
    return result;
  }
  while ( v17 != -4096 )
  {
    if ( !v12 && v17 == -8192 )
      v12 = v16;
    v15 = (v6 - 1) & (v14 + v15);
    v16 = (__int64 *)(v9 + 32LL * v15);
    v17 = *v16;
    if ( a3 == *v16 )
      goto LABEL_3;
    ++v14;
  }
  if ( !v12 )
    v12 = v16;
  v29 = *(_DWORD *)(v5 + 152);
  ++*(_QWORD *)(v5 + 136);
  v30 = v29 + 1;
  if ( 4 * (v29 + 1) >= 3 * v6 )
  {
LABEL_25:
    sub_124E280(v7, 2 * v6);
    v32 = *(_DWORD *)(v5 + 160);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(v5 + 144);
      v35 = (v32 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v30 = *(_DWORD *)(v5 + 152) + 1;
      v12 = (__int64 *)(v34 + 32LL * v35);
      v36 = *v12;
      if ( a3 != *v12 )
      {
        v37 = 1;
        v38 = 0;
        while ( v36 != -4096 )
        {
          if ( !v38 && v36 == -8192 )
            v38 = v12;
          v35 = v33 & (v37 + v35);
          v12 = (__int64 *)(v34 + 32LL * v35);
          v36 = *v12;
          if ( a3 == *v12 )
            goto LABEL_20;
          ++v37;
        }
        if ( v38 )
          v12 = v38;
      }
      goto LABEL_20;
    }
    goto LABEL_49;
  }
  if ( v6 - *(_DWORD *)(v5 + 156) - v30 <= v6 >> 3 )
  {
    sub_124E280(v7, v6);
    v39 = *(_DWORD *)(v5 + 160);
    if ( v39 )
    {
      v40 = v39 - 1;
      v41 = *(_QWORD *)(v5 + 144);
      v42 = 0;
      v43 = v40 & v13;
      v44 = 1;
      v30 = *(_DWORD *)(v5 + 152) + 1;
      v12 = (__int64 *)(v41 + 32LL * v43);
      v45 = *v12;
      if ( a3 != *v12 )
      {
        while ( v45 != -4096 )
        {
          if ( !v42 && v45 == -8192 )
            v42 = v12;
          v43 = v40 & (v44 + v43);
          v12 = (__int64 *)(v41 + 32LL * v43);
          v45 = *v12;
          if ( a3 == *v12 )
            goto LABEL_20;
          ++v44;
        }
        if ( v42 )
          v12 = v42;
      }
      goto LABEL_20;
    }
LABEL_49:
    ++*(_DWORD *)(v5 + 152);
    BUG();
  }
LABEL_20:
  *(_DWORD *)(v5 + 152) = v30;
  if ( *v12 != -4096 )
    --*(_DWORD *)(v5 + 156);
  *v12 = a3;
  v12[1] = 0;
  v12[2] = 0;
  v12[3] = 0;
  return 0;
}
