// Function: sub_1170890
// Address: 0x1170890
//
__int64 __fastcall sub_1170890(__int64 *a1, __int64 a2)
{
  __int64 v2; // r9
  char v3; // cl
  __int64 v4; // r8
  int v5; // eax
  unsigned int v6; // edx
  __int64 *v7; // rbx
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 v10; // r12
  char v11; // cl
  unsigned int v12; // esi
  __int64 v13; // r9
  int v14; // esi
  __int64 v15; // r10
  unsigned int v16; // edx
  __int64 v17; // rax
  __int64 v18; // r11
  __int64 v19; // r8
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rdx
  unsigned int v24; // eax
  __int64 v25; // r8
  int v26; // edx
  unsigned int v27; // edi
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 v30; // rbx
  int v31; // r11d
  int v32; // r14d
  __int64 v33; // rsi
  int v34; // edx
  __int64 v35; // rcx
  unsigned int v36; // eax
  __int64 v37; // rdi
  __int64 v38; // rsi
  int v39; // edx
  __int64 v40; // rcx
  unsigned int v41; // eax
  __int64 v42; // rdi
  int v43; // r10d
  __int64 v44; // r9
  int v45; // edx
  int v46; // edx
  int v47; // r10d
  __int64 v48[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v49[6]; // [rsp+10h] [rbp-30h] BYREF

  v2 = *a1;
  v3 = *(_BYTE *)(*a1 + 8) & 1;
  if ( v3 )
  {
    v4 = v2 + 16;
    v5 = 7;
  }
  else
  {
    v29 = *(unsigned int *)(v2 + 24);
    v4 = *(_QWORD *)(v2 + 16);
    if ( !(_DWORD)v29 )
      goto LABEL_25;
    v5 = v29 - 1;
  }
  v6 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( a2 == *v7 )
    goto LABEL_4;
  v31 = 1;
  while ( v8 != -4096 )
  {
    v6 = v5 & (v31 + v6);
    v7 = (__int64 *)(v4 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
      goto LABEL_4;
    ++v31;
  }
  if ( v3 )
  {
    v30 = 128;
    goto LABEL_26;
  }
  v29 = *(unsigned int *)(v2 + 24);
LABEL_25:
  v30 = 16 * v29;
LABEL_26:
  v7 = (__int64 *)(v4 + v30);
LABEL_4:
  v9 = 128;
  if ( !v3 )
    v9 = 16LL * *(unsigned int *)(v2 + 24);
  if ( v7 == (__int64 *)(v4 + v9) )
    return 0;
  v10 = a1[1];
  v11 = *(_BYTE *)(v10 + 8) & 1;
  if ( v11 )
  {
    v13 = v10 + 16;
    v14 = 7;
  }
  else
  {
    v12 = *(_DWORD *)(v10 + 24);
    v13 = *(_QWORD *)(v10 + 16);
    if ( !v12 )
    {
      v24 = *(_DWORD *)(v10 + 8);
      ++*(_QWORD *)v10;
      v25 = 0;
      v26 = (v24 >> 1) + 1;
      goto LABEL_14;
    }
    v14 = v12 - 1;
  }
  v15 = v7[1];
  v16 = v14 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  v17 = v13 + 16LL * v16;
  v18 = *(_QWORD *)v17;
  if ( v15 != *(_QWORD *)v17 )
  {
    v32 = 1;
    v25 = 0;
    while ( v18 != -4096 )
    {
      if ( v18 == -8192 && !v25 )
        v25 = v17;
      v16 = v14 & (v32 + v16);
      v17 = v13 + 16LL * v16;
      v18 = *(_QWORD *)v17;
      if ( v15 == *(_QWORD *)v17 )
        goto LABEL_11;
      ++v32;
    }
    v27 = 24;
    v12 = 8;
    if ( !v25 )
      v25 = v17;
    v24 = *(_DWORD *)(v10 + 8);
    ++*(_QWORD *)v10;
    v26 = (v24 >> 1) + 1;
    if ( v11 )
    {
LABEL_15:
      if ( 4 * v26 < v27 )
      {
        if ( v12 - *(_DWORD *)(v10 + 12) - v26 > v12 >> 3 )
        {
LABEL_17:
          *(_DWORD *)(v10 + 8) = (2 * (v24 >> 1) + 2) | v24 & 1;
          if ( *(_QWORD *)v25 != -4096 )
            --*(_DWORD *)(v10 + 12);
          v28 = v7[1];
          *(_DWORD *)(v25 + 8) = 0;
          *(_QWORD *)v25 = v28;
          return 0;
        }
        sub_FB9E50(v10, v12);
        if ( (*(_BYTE *)(v10 + 8) & 1) != 0 )
        {
          v38 = v10 + 16;
          v39 = 7;
          goto LABEL_43;
        }
        v46 = *(_DWORD *)(v10 + 24);
        v38 = *(_QWORD *)(v10 + 16);
        if ( v46 )
        {
          v39 = v46 - 1;
LABEL_43:
          v40 = v7[1];
          v41 = v39 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          v25 = v38 + 16LL * v41;
          v42 = *(_QWORD *)v25;
          if ( *(_QWORD *)v25 != v40 )
          {
            v43 = 1;
            v44 = 0;
            while ( v42 != -4096 )
            {
              if ( !v44 && v42 == -8192 )
                v44 = v25;
              v41 = v39 & (v43 + v41);
              v25 = v38 + 16LL * v41;
              v42 = *(_QWORD *)v25;
              if ( v40 == *(_QWORD *)v25 )
                goto LABEL_40;
              ++v43;
            }
LABEL_46:
            if ( v44 )
              v25 = v44;
            goto LABEL_40;
          }
          goto LABEL_40;
        }
LABEL_71:
        *(_DWORD *)(v10 + 8) = (2 * (*(_DWORD *)(v10 + 8) >> 1) + 2) | *(_DWORD *)(v10 + 8) & 1;
        BUG();
      }
      sub_FB9E50(v10, 2 * v12);
      if ( (*(_BYTE *)(v10 + 8) & 1) != 0 )
      {
        v33 = v10 + 16;
        v34 = 7;
      }
      else
      {
        v45 = *(_DWORD *)(v10 + 24);
        v33 = *(_QWORD *)(v10 + 16);
        if ( !v45 )
          goto LABEL_71;
        v34 = v45 - 1;
      }
      v35 = v7[1];
      v36 = v34 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v25 = v33 + 16LL * v36;
      v37 = *(_QWORD *)v25;
      if ( *(_QWORD *)v25 != v35 )
      {
        v47 = 1;
        v44 = 0;
        while ( v37 != -4096 )
        {
          if ( v37 == -8192 && !v44 )
            v44 = v25;
          v36 = v34 & (v47 + v36);
          v25 = v33 + 16LL * v36;
          v37 = *(_QWORD *)v25;
          if ( v35 == *(_QWORD *)v25 )
            goto LABEL_40;
          ++v47;
        }
        goto LABEL_46;
      }
LABEL_40:
      v24 = *(_DWORD *)(v10 + 8);
      goto LABEL_17;
    }
    v12 = *(_DWORD *)(v10 + 24);
LABEL_14:
    v27 = 3 * v12;
    goto LABEL_15;
  }
LABEL_11:
  if ( *(_DWORD *)(v17 + 8) == 1 )
  {
    v19 = a1[2];
    v20 = *(_QWORD *)a1[5];
    v49[0] = *(_QWORD *)a1[4];
    v21 = (__int64 *)a1[3];
    v49[1] = v20;
    v22 = *v21;
    v48[1] = v7[1];
    v48[0] = v22;
    return sub_B1A0F0(v19, v48, v49);
  }
  return 0;
}
