// Function: sub_1DE1E40
// Address: 0x1de1e40
//
__int64 __fastcall sub_1DE1E40(__int64 a1, __int64 a2, __int64 a3)
{
  int *v5; // rbx
  int v6; // edx
  char v7; // dl
  __int64 v8; // r10
  int v9; // esi
  int v10; // ecx
  unsigned int v11; // edi
  __int64 v12; // rax
  int v13; // r9d
  __int64 result; // rax
  unsigned int v15; // esi
  unsigned int v16; // eax
  __int64 v17; // r8
  int v18; // ecx
  unsigned int v19; // edi
  int v20; // edx
  unsigned __int64 v21; // r14
  int v22; // r11d
  __int64 v23; // rdi
  int v24; // ecx
  int v25; // eax
  unsigned int v26; // esi
  __int64 v27; // rdi
  int v28; // esi
  int v29; // ecx
  unsigned int v30; // eax
  int v31; // r10d
  __int64 v32; // r9
  int v33; // ecx
  int v34; // esi
  __int64 v35; // rbx
  int v36; // r10d
  int *v37; // [rsp+8h] [rbp-48h]
  _DWORD v38[13]; // [rsp+1Ch] [rbp-34h] BYREF

  if ( a2 )
  {
    sub_1376790(a1, a2);
    v5 = *(int **)(a2 + 96);
    v37 = &v5[*(unsigned int *)(a2 + 104)];
    while ( v37 != v5 )
    {
      v6 = *v5++;
      v38[0] = v6;
      sub_1DDEB40(a1, v38, a2, a3);
    }
  }
  else
  {
    sub_1376580(a1);
    LODWORD(v21) = 0;
    if ( *(_QWORD *)(*(_QWORD *)a1 + 64LL) != *(_QWORD *)(*(_QWORD *)a1 + 72LL) )
    {
      do
      {
        v38[0] = v21;
        sub_1DDEB40(a1, v38, 0, a3);
        v21 = (unsigned int)(v21 + 1);
      }
      while ( v21 < 0xAAAAAAAAAAAAAAABLL
                  * ((__int64)(*(_QWORD *)(*(_QWORD *)a1 + 72LL) - *(_QWORD *)(*(_QWORD *)a1 + 64LL)) >> 3) );
    }
  }
  v7 = *(_BYTE *)(a1 + 56) & 1;
  if ( v7 )
  {
    v8 = a1 + 64;
    v9 = 3;
  }
  else
  {
    v15 = *(_DWORD *)(a1 + 72);
    v8 = *(_QWORD *)(a1 + 64);
    if ( !v15 )
    {
      v16 = *(_DWORD *)(a1 + 56);
      ++*(_QWORD *)(a1 + 48);
      v17 = 0;
      v18 = (v16 >> 1) + 1;
LABEL_12:
      v19 = 3 * v15;
      goto LABEL_13;
    }
    v9 = v15 - 1;
  }
  v10 = *(_DWORD *)(a1 + 8);
  v11 = v9 & (37 * v10);
  v12 = v8 + 16LL * v11;
  v13 = *(_DWORD *)v12;
  if ( v10 == *(_DWORD *)v12 )
  {
    result = *(_QWORD *)(v12 + 8);
    goto LABEL_8;
  }
  v22 = 1;
  v17 = 0;
  while ( v13 != -1 )
  {
    if ( v17 || v13 != -2 )
      v12 = v17;
    v11 = v9 & (v22 + v11);
    v35 = v8 + 16LL * v11;
    v13 = *(_DWORD *)v35;
    if ( v10 == *(_DWORD *)v35 )
    {
      result = *(_QWORD *)(v35 + 8);
      goto LABEL_8;
    }
    ++v22;
    v17 = v12;
    v12 = v8 + 16LL * v11;
  }
  v19 = 12;
  v15 = 4;
  if ( !v17 )
    v17 = v12;
  v16 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 48);
  v18 = (v16 >> 1) + 1;
  if ( !v7 )
  {
    v15 = *(_DWORD *)(a1 + 72);
    goto LABEL_12;
  }
LABEL_13:
  if ( v19 <= 4 * v18 )
  {
    sub_136EE70(a1 + 48, 2 * v15);
    if ( (*(_BYTE *)(a1 + 56) & 1) != 0 )
    {
      v23 = a1 + 64;
      v24 = 3;
    }
    else
    {
      v33 = *(_DWORD *)(a1 + 72);
      v23 = *(_QWORD *)(a1 + 64);
      if ( !v33 )
        goto LABEL_63;
      v24 = v33 - 1;
    }
    v25 = *(_DWORD *)(a1 + 8);
    v26 = v24 & (37 * v25);
    v17 = v23 + 16LL * v26;
    v20 = *(_DWORD *)v17;
    if ( v25 == *(_DWORD *)v17 )
      goto LABEL_31;
    v36 = 1;
    v32 = 0;
    while ( v20 != -1 )
    {
      if ( !v32 && v20 == -2 )
        v32 = v17;
      v26 = v24 & (v36 + v26);
      v17 = v23 + 16LL * v26;
      v20 = *(_DWORD *)v17;
      if ( v25 == *(_DWORD *)v17 )
        goto LABEL_31;
      ++v36;
    }
    v20 = *(_DWORD *)(a1 + 8);
    if ( !v32 )
      goto LABEL_31;
LABEL_38:
    v17 = v32;
LABEL_31:
    v16 = *(_DWORD *)(a1 + 56);
    goto LABEL_16;
  }
  if ( v15 - *(_DWORD *)(a1 + 60) - v18 <= v15 >> 3 )
  {
    sub_136EE70(a1 + 48, v15);
    if ( (*(_BYTE *)(a1 + 56) & 1) != 0 )
    {
      v27 = a1 + 64;
      v28 = 3;
LABEL_34:
      v29 = *(_DWORD *)(a1 + 8);
      v30 = v28 & (37 * v29);
      v17 = v27 + 16LL * v30;
      v20 = *(_DWORD *)v17;
      if ( v29 == *(_DWORD *)v17 )
        goto LABEL_31;
      v31 = 1;
      v32 = 0;
      while ( v20 != -1 )
      {
        if ( v20 == -2 && !v32 )
          v32 = v17;
        v30 = v28 & (v31 + v30);
        v17 = v27 + 16LL * v30;
        v20 = *(_DWORD *)v17;
        if ( v29 == *(_DWORD *)v17 )
          goto LABEL_31;
        ++v31;
      }
      v20 = *(_DWORD *)(a1 + 8);
      if ( !v32 )
        goto LABEL_31;
      goto LABEL_38;
    }
    v34 = *(_DWORD *)(a1 + 72);
    v27 = *(_QWORD *)(a1 + 64);
    if ( v34 )
    {
      v28 = v34 - 1;
      goto LABEL_34;
    }
LABEL_63:
    *(_DWORD *)(a1 + 56) = (2 * (*(_DWORD *)(a1 + 56) >> 1) + 2) | *(_DWORD *)(a1 + 56) & 1;
    BUG();
  }
  v20 = *(_DWORD *)(a1 + 8);
LABEL_16:
  *(_DWORD *)(a1 + 56) = (2 * (v16 >> 1) + 2) | v16 & 1;
  if ( *(_DWORD *)v17 != -1 )
    --*(_DWORD *)(a1 + 60);
  *(_DWORD *)v17 = v20;
  result = 0;
  *(_QWORD *)(v17 + 8) = 0;
LABEL_8:
  *(_QWORD *)(a1 + 16) = result;
  return result;
}
