// Function: sub_38D3840
// Address: 0x38d3840
//
__int64 __fastcall sub_38D3840(__int64 a1, __int64 a2)
{
  int v3; // r13d
  _BYTE *v5; // rax
  int v6; // ecx
  __int64 *v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // r12
  unsigned int v11; // esi
  int v12; // r14d
  __int64 v13; // r9
  unsigned int v14; // r8d
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  int v18; // eax
  int v19; // r8d
  __int64 v20; // rsi
  __int64 v21; // rax
  int v22; // edi
  __int64 *v23; // rdx
  __int64 v24; // r9
  int v25; // eax
  int v26; // eax
  __int64 v27; // rsi
  __int64 *v28; // r9
  __int64 v29; // r15
  __int64 v30; // r8
  __int64 *v31; // r10
  int v32; // [rsp+Ch] [rbp-84h]
  int v33; // [rsp+Ch] [rbp-84h]
  int v34; // [rsp+Ch] [rbp-84h]
  _QWORD *v35; // [rsp+10h] [rbp-80h] BYREF
  __int64 v36; // [rsp+18h] [rbp-78h]
  const char *v37; // [rsp+20h] [rbp-70h] BYREF
  char v38; // [rsp+30h] [rbp-60h]
  char v39; // [rsp+31h] [rbp-5Fh]
  _QWORD *v40; // [rsp+40h] [rbp-50h] BYREF
  __int16 v41; // [rsp+50h] [rbp-40h]

  v3 = *(_DWORD *)(a1 + 680);
  if ( v3 != 1 )
    return *(_QWORD *)(a1 + 416);
  v5 = *(_BYTE **)(a2 + 184);
  v35 = 0;
  v6 = 128;
  v36 = 0;
  if ( v5 )
  {
    if ( (*v5 & 4) != 0 )
    {
      v7 = (__int64 *)*((_QWORD *)v5 - 1);
      v8 = *v7;
      v9 = v7 + 2;
    }
    else
    {
      v8 = 0;
      v9 = 0;
    }
    v35 = v9;
    v6 = 640;
    v36 = v8;
  }
  v10 = *(_QWORD *)(a2 + 8);
  v11 = *(_DWORD *)(a1 + 448);
  v12 = *(_DWORD *)(a1 + 440);
  if ( !v11 )
  {
    ++*(_QWORD *)(a1 + 424);
    goto LABEL_13;
  }
  v13 = *(_QWORD *)(a1 + 432);
  v14 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v15 = (__int64 *)(v13 + 16LL * v14);
  v16 = *v15;
  if ( v10 != *v15 )
  {
    v33 = 1;
    v23 = 0;
    while ( v16 != -8 )
    {
      if ( !v23 && v16 == -16 )
        v23 = v15;
      v14 = (v11 - 1) & (v33 + v14);
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( v10 == *v15 )
        goto LABEL_9;
      ++v33;
    }
    v22 = v12 + 1;
    if ( !v23 )
      v23 = v15;
    ++*(_QWORD *)(a1 + 424);
    if ( 4 * v22 < 3 * v11 )
    {
      if ( v11 - *(_DWORD *)(a1 + 444) - v22 > v11 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(a1 + 440) = v22;
        if ( *v23 != -8 )
          --*(_DWORD *)(a1 + 444);
        *v23 = v10;
        *((_DWORD *)v23 + 2) = v12;
        goto LABEL_10;
      }
      v34 = v6;
      sub_211A5E0(a1 + 424, v11);
      v25 = *(_DWORD *)(a1 + 448);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *(_QWORD *)(a1 + 432);
        v28 = 0;
        LODWORD(v29) = v26 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v22 = *(_DWORD *)(a1 + 440) + 1;
        v6 = v34;
        v23 = (__int64 *)(v27 + 16LL * (unsigned int)v29);
        v30 = *v23;
        if ( v10 != *v23 )
        {
          while ( v30 != -8 )
          {
            if ( !v28 && v30 == -16 )
              v28 = v23;
            v29 = v26 & (unsigned int)(v29 + v3);
            v23 = (__int64 *)(v27 + 16 * v29);
            v30 = *v23;
            if ( v10 == *v23 )
              goto LABEL_15;
            ++v3;
          }
          if ( v28 )
            v23 = v28;
        }
        goto LABEL_15;
      }
LABEL_50:
      ++*(_DWORD *)(a1 + 440);
      BUG();
    }
LABEL_13:
    v32 = v6;
    sub_211A5E0(a1 + 424, 2 * v11);
    v18 = *(_DWORD *)(a1 + 448);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 432);
      LODWORD(v21) = (v18 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v22 = *(_DWORD *)(a1 + 440) + 1;
      v6 = v32;
      v23 = (__int64 *)(v20 + 16LL * (unsigned int)v21);
      v24 = *v23;
      if ( v10 != *v23 )
      {
        v31 = 0;
        while ( v24 != -8 )
        {
          if ( !v31 && v24 == -16 )
            v31 = v23;
          v21 = v19 & (unsigned int)(v21 + v3);
          v23 = (__int64 *)(v20 + 16 * v21);
          v24 = *v23;
          if ( v10 == *v23 )
            goto LABEL_15;
          ++v3;
        }
        if ( v31 )
          v23 = v31;
      }
      goto LABEL_15;
    }
    goto LABEL_50;
  }
LABEL_9:
  v12 = *((_DWORD *)v15 + 2);
LABEL_10:
  v17 = *(_QWORD *)(a1 + 688);
  v41 = 261;
  v40 = &v35;
  v39 = 1;
  v37 = ".stack_sizes";
  v38 = 3;
  return sub_38C3B80(v17, (__int64)&v37, 1, v6, 0, (__int64)&v40, v12, v10);
}
