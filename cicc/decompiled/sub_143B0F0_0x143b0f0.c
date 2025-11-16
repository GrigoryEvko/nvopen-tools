// Function: sub_143B0F0
// Address: 0x143b0f0
//
bool __fastcall sub_143B0F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // r13
  __int64 v9; // r8
  __int64 v11; // rdi
  int v12; // esi
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r10
  int v16; // r14d
  char v17; // cl
  __int64 v18; // r12
  int v19; // ecx
  unsigned int v20; // esi
  unsigned int v21; // edx
  int v22; // edi
  unsigned int v23; // r10d
  __int64 v25; // rsi
  int v26; // ecx
  unsigned int v27; // edx
  __int64 v28; // rdi
  __int64 v29; // rsi
  int v30; // ecx
  unsigned int v31; // edx
  __int64 v32; // rdi
  __int64 v33; // r10
  int v34; // ecx
  int v35; // ecx
  int v36; // [rsp+10h] [rbp-40h]
  __int64 v37; // [rsp+10h] [rbp-40h]
  __int64 v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+18h] [rbp-38h]
  __int64 v40; // [rsp+18h] [rbp-38h]
  __int64 v41; // [rsp+18h] [rbp-38h]
  int v42; // [rsp+18h] [rbp-38h]
  int v43; // [rsp+18h] [rbp-38h]

  v5 = *(_QWORD *)(a1 + 544);
  v6 = *(_QWORD *)(a1 + 528);
  v7 = v5 + 40;
  if ( v6 == v5 + 40 )
    v8 = *(_QWORD *)(v5 + 48);
  else
    v8 = *(_QWORD *)(v6 + 8);
  if ( v8 == v7 )
  {
    v18 = 0;
    goto LABEL_23;
  }
  v9 = a3;
  while ( 1 )
  {
    v16 = *(_DWORD *)(a1 + 536);
    v17 = *(_BYTE *)(a1 + 8);
    v18 = v8 - 24;
    if ( !v8 )
      v18 = 0;
    *(_DWORD *)(a1 + 536) = v16 + 1;
    v19 = v17 & 1;
    if ( v19 )
    {
      v11 = a1 + 16;
      v12 = 31;
    }
    else
    {
      v20 = *(_DWORD *)(a1 + 24);
      v11 = *(_QWORD *)(a1 + 16);
      if ( !v20 )
      {
        v21 = *(_DWORD *)(a1 + 8);
        ++*(_QWORD *)a1;
        v14 = 0;
        v22 = (v21 >> 1) + 1;
LABEL_16:
        v23 = 3 * v20;
        goto LABEL_17;
      }
      v12 = v20 - 1;
    }
    v13 = v12 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v14 = v11 + 16LL * v13;
    v15 = *(_QWORD *)v14;
    if ( v18 == *(_QWORD *)v14 )
    {
LABEL_7:
      *(_DWORD *)(v14 + 8) = v16;
      if ( a2 == v18 )
        break;
      goto LABEL_8;
    }
    v36 = 1;
    v39 = 0;
    while ( v15 != -8 )
    {
      if ( !v39 )
      {
        if ( v15 != -16 )
          v14 = 0;
        v39 = v14;
      }
      v13 = v12 & (v36 + v13);
      v14 = v11 + 16LL * v13;
      v15 = *(_QWORD *)v14;
      if ( v18 == *(_QWORD *)v14 )
        goto LABEL_7;
      ++v36;
    }
    v21 = *(_DWORD *)(a1 + 8);
    v23 = 96;
    if ( v39 )
      v14 = v39;
    ++*(_QWORD *)a1;
    v20 = 32;
    v22 = (v21 >> 1) + 1;
    if ( !(_BYTE)v19 )
    {
      v20 = *(_DWORD *)(a1 + 24);
      goto LABEL_16;
    }
LABEL_17:
    if ( v23 <= 4 * v22 )
    {
      v37 = v9;
      v40 = v7;
      sub_143AD00(a1, 2 * v20);
      v7 = v40;
      v9 = v37;
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v25 = a1 + 16;
        v26 = 31;
      }
      else
      {
        v34 = *(_DWORD *)(a1 + 24);
        v25 = *(_QWORD *)(a1 + 16);
        if ( !v34 )
          goto LABEL_65;
        v26 = v34 - 1;
      }
      v27 = v26 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v14 = v25 + 16LL * v27;
      v28 = *(_QWORD *)v14;
      if ( v18 == *(_QWORD *)v14 )
        goto LABEL_34;
      v43 = 1;
      v33 = 0;
      while ( v28 != -8 )
      {
        if ( v28 == -16 && !v33 )
          v33 = v14;
        v27 = v26 & (v43 + v27);
        v14 = v25 + 16LL * v27;
        v28 = *(_QWORD *)v14;
        if ( v18 == *(_QWORD *)v14 )
          goto LABEL_34;
        ++v43;
      }
      goto LABEL_40;
    }
    if ( v20 - *(_DWORD *)(a1 + 12) - v22 > v20 >> 3 )
      goto LABEL_19;
    v38 = v9;
    v41 = v7;
    sub_143AD00(a1, v20);
    v7 = v41;
    v9 = v38;
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v29 = a1 + 16;
      v30 = 31;
    }
    else
    {
      v35 = *(_DWORD *)(a1 + 24);
      v29 = *(_QWORD *)(a1 + 16);
      if ( !v35 )
      {
LABEL_65:
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        BUG();
      }
      v30 = v35 - 1;
    }
    v31 = v30 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v14 = v29 + 16LL * v31;
    v32 = *(_QWORD *)v14;
    if ( v18 != *(_QWORD *)v14 )
    {
      v42 = 1;
      v33 = 0;
      while ( v32 != -8 )
      {
        if ( !v33 && v32 == -16 )
          v33 = v14;
        v31 = v30 & (v42 + v31);
        v14 = v29 + 16LL * v31;
        v32 = *(_QWORD *)v14;
        if ( v18 == *(_QWORD *)v14 )
          goto LABEL_34;
        ++v42;
      }
LABEL_40:
      if ( v33 )
        v14 = v33;
    }
LABEL_34:
    v21 = *(_DWORD *)(a1 + 8);
LABEL_19:
    *(_DWORD *)(a1 + 8) = (2 * (v21 >> 1) + 2) | v21 & 1;
    if ( *(_QWORD *)v14 != -8 )
      --*(_DWORD *)(a1 + 12);
    *(_DWORD *)(v14 + 8) = 0;
    *(_QWORD *)v14 = v18;
    *(_DWORD *)(v14 + 8) = v16;
    if ( a2 == v18 )
      break;
LABEL_8:
    if ( v9 != v18 )
    {
      v8 = *(_QWORD *)(v8 + 8);
      if ( v7 != v8 )
        continue;
    }
    break;
  }
  a3 = v9;
LABEL_23:
  *(_QWORD *)(a1 + 528) = v8;
  return a3 != v18;
}
