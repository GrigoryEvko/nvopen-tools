// Function: sub_13C0C50
// Address: 0x13c0c50
//
__int64 __fastcall sub_13C0C50(_QWORD *a1)
{
  __int64 v2; // r13
  unsigned __int8 v3; // al
  __int64 v4; // rbx
  int v5; // eax
  int v6; // ecx
  int v7; // edi
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 *v10; // r14
  __int64 v11; // rsi
  unsigned __int64 v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rax
  int v15; // eax
  int v16; // esi
  __int64 v17; // rcx
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rbx
  _QWORD *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  int v32; // eax
  int v33; // r8d
  _QWORD *v34; // rdx
  _QWORD *v35; // rdx
  _QWORD *v36; // rcx
  _QWORD *v37; // rax
  unsigned int v38; // edi
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // r9
  int v41; // r8d
  unsigned int v42; // r10d
  __int64 *v43; // rsi
  __int64 v44; // r11
  unsigned int v45; // esi
  int v46; // r8d
  __int64 v47; // rdx
  __int64 v48; // rcx
  int v49; // esi
  int v50; // ebx
  _QWORD *v51; // rdx
  _QWORD *v52; // rcx
  _QWORD *v53; // rax

  v2 = a1[3];
  v3 = *(_BYTE *)(v2 + 16);
  if ( !v3 )
  {
    v4 = a1[4];
    v5 = *(_DWORD *)(v4 + 288);
    if ( !v5 )
      goto LABEL_7;
    v6 = v5 - 1;
    v7 = 1;
    v8 = *(_QWORD *)(v4 + 272);
    v9 = (v5 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v10 = (__int64 *)(v8 + 16LL * v9);
    v11 = *v10;
    if ( v2 != *v10 )
    {
      while ( v11 != -8 )
      {
        v9 = v6 & (v7 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( v2 == *v10 )
          goto LABEL_4;
        ++v7;
      }
      goto LABEL_7;
    }
LABEL_4:
    v12 = v10[1] & 0xFFFFFFFFFFFFFFF8LL;
    if ( v12 )
    {
      if ( (*(_BYTE *)(v12 + 8) & 1) == 0 )
        j___libc_free_0(*(_QWORD *)(v12 + 16));
      j_j___libc_free_0(v12, 272);
    }
    *v10 = -16;
    --*(_DWORD *)(v4 + 280);
    ++*(_DWORD *)(v4 + 284);
    v3 = *(_BYTE *)(v2 + 16);
  }
  v4 = a1[4];
  if ( v3 > 3u )
    goto LABEL_11;
LABEL_7:
  v13 = *(_QWORD **)(v4 + 32);
  if ( *(_QWORD **)(v4 + 40) == v13 )
  {
    v26 = &v13[*(unsigned int *)(v4 + 52)];
    if ( v13 == v26 )
    {
LABEL_35:
      v13 = v26;
    }
    else
    {
      while ( v2 != *v13 )
      {
        if ( v26 == ++v13 )
          goto LABEL_35;
      }
    }
  }
  else
  {
    v13 = (_QWORD *)sub_16CC9F0(v4 + 24, v2);
    if ( v2 == *v13 )
    {
      v30 = *(_QWORD *)(v4 + 40);
      if ( v30 == *(_QWORD *)(v4 + 32) )
        v31 = *(unsigned int *)(v4 + 52);
      else
        v31 = *(unsigned int *)(v4 + 48);
      v26 = (_QWORD *)(v30 + 8 * v31);
    }
    else
    {
      v14 = *(_QWORD *)(v4 + 40);
      if ( v14 != *(_QWORD *)(v4 + 32) )
      {
LABEL_10:
        v4 = a1[4];
        goto LABEL_11;
      }
      v13 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(v4 + 52));
      v26 = v13;
    }
  }
  if ( v26 == v13 )
    goto LABEL_10;
  *v13 = -2;
  ++*(_DWORD *)(v4 + 56);
  v27 = a1[4];
  v28 = *(_QWORD **)(v27 + 136);
  if ( *(_QWORD **)(v27 + 144) == v28 )
  {
    v34 = &v28[*(unsigned int *)(v27 + 156)];
    if ( v28 == v34 )
    {
LABEL_78:
      v28 = v34;
    }
    else
    {
      while ( v2 != *v28 )
      {
        if ( v34 == ++v28 )
          goto LABEL_78;
      }
    }
  }
  else
  {
    v28 = (_QWORD *)sub_16CC9F0(v27 + 128, v2);
    if ( v2 == *v28 )
    {
      v47 = *(_QWORD *)(v27 + 144);
      if ( v47 == *(_QWORD *)(v27 + 136) )
        v48 = *(unsigned int *)(v27 + 156);
      else
        v48 = *(unsigned int *)(v27 + 152);
      v34 = (_QWORD *)(v47 + 8 * v48);
    }
    else
    {
      v29 = *(_QWORD *)(v27 + 144);
      if ( v29 != *(_QWORD *)(v27 + 136) )
      {
LABEL_31:
        v4 = a1[4];
        goto LABEL_49;
      }
      v28 = (_QWORD *)(v29 + 8LL * *(unsigned int *)(v27 + 156));
      v34 = v28;
    }
  }
  if ( v28 == v34 )
    goto LABEL_31;
  *v28 = -2;
  ++*(_DWORD *)(v27 + 160);
  v4 = a1[4];
  if ( *(_DWORD *)(v4 + 248) )
  {
    v51 = *(_QWORD **)(v4 + 240);
    v52 = &v51[2 * *(unsigned int *)(v4 + 256)];
    if ( v51 != v52 )
    {
      while ( 1 )
      {
        v53 = v51;
        if ( *v51 != -16 && *v51 != -8 )
          break;
        v51 += 2;
        if ( v52 == v51 )
          goto LABEL_49;
      }
      while ( v52 != v53 )
      {
        if ( v53[1] == v2 )
        {
          *v53 = -16;
          --*(_DWORD *)(v4 + 248);
          ++*(_DWORD *)(v4 + 252);
          v4 = a1[4];
        }
        v53 += 2;
        if ( v53 == v52 )
          break;
        while ( *v53 == -16 || *v53 == -8 )
        {
          v53 += 2;
          if ( v52 == v53 )
            goto LABEL_49;
        }
      }
    }
  }
LABEL_49:
  if ( *(_DWORD *)(v4 + 280) )
  {
    v35 = *(_QWORD **)(v4 + 272);
    v36 = &v35[2 * *(unsigned int *)(v4 + 288)];
    if ( v35 != v36 )
    {
      while ( 1 )
      {
        v37 = v35;
        if ( *v35 != -16 && *v35 != -8 )
          break;
        v35 += 2;
        if ( v36 == v35 )
          goto LABEL_11;
      }
      if ( v36 != v35 )
      {
        v38 = ((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4);
        v39 = v35[1] & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v37[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          goto LABEL_65;
        while ( 1 )
        {
          while ( 1 )
          {
            do
            {
              v37 += 2;
              if ( v37 == v36 )
                goto LABEL_10;
              while ( *v37 == -8 || *v37 == -16 )
              {
                v37 += 2;
                if ( v36 == v37 )
                  goto LABEL_10;
              }
              if ( v36 == v37 )
                goto LABEL_10;
              v39 = v37[1] & 0xFFFFFFFFFFFFFFF8LL;
            }
            while ( !v39 );
LABEL_65:
            if ( (*(_BYTE *)(v39 + 8) & 1) == 0 )
              break;
            v40 = v39 + 16;
            v41 = 15;
LABEL_67:
            v42 = v41 & v38;
            v43 = (__int64 *)(v40 + 16LL * (v41 & v38));
            v44 = *v43;
            if ( v2 == *v43 )
            {
LABEL_68:
              *v43 = -16;
              v45 = *(_DWORD *)(v39 + 8);
              ++*(_DWORD *)(v39 + 12);
              *(_DWORD *)(v39 + 8) = (2 * (v45 >> 1) - 2) | v45 & 1;
            }
            else
            {
              v49 = 1;
              while ( v44 != -8 )
              {
                v50 = v49 + 1;
                v42 = v41 & (v49 + v42);
                v43 = (__int64 *)(v40 + 16LL * v42);
                v44 = *v43;
                if ( v2 == *v43 )
                  goto LABEL_68;
                v49 = v50;
              }
            }
          }
          v46 = *(_DWORD *)(v39 + 24);
          v40 = *(_QWORD *)(v39 + 16);
          if ( v46 )
          {
            v41 = v46 - 1;
            goto LABEL_67;
          }
        }
      }
    }
  }
LABEL_11:
  v15 = *(_DWORD *)(v4 + 256);
  if ( v15 )
  {
    v16 = v15 - 1;
    v17 = *(_QWORD *)(v4 + 240);
    v18 = (v15 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v19 = (__int64 *)(v17 + 16LL * v18);
    v20 = *v19;
    if ( v2 == *v19 )
    {
LABEL_13:
      *v19 = -16;
      --*(_DWORD *)(v4 + 248);
      ++*(_DWORD *)(v4 + 252);
    }
    else
    {
      v32 = 1;
      while ( v20 != -8 )
      {
        v33 = v32 + 1;
        v18 = v16 & (v32 + v18);
        v19 = (__int64 *)(v17 + 16LL * v18);
        v20 = *v19;
        if ( v2 == *v19 )
          goto LABEL_13;
        v32 = v33;
      }
    }
  }
  v21 = a1[3];
  if ( v21 )
  {
    if ( v21 != -16 && v21 != -8 )
      sub_1649B30(a1 + 1);
    a1[3] = 0;
  }
  v22 = a1[4];
  v23 = a1[5];
  --*(_QWORD *)(v22 + 344);
  sub_2208CA0(v23);
  *(_QWORD *)(v23 + 16) = &unk_49EE2B0;
  v24 = *(_QWORD *)(v23 + 40);
  if ( v24 != 0 && v24 != -8 && v24 != -16 )
    sub_1649B30(v23 + 24);
  return j_j___libc_free_0(v23, 64);
}
