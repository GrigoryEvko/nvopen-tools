// Function: sub_D1A870
// Address: 0xd1a870
//
__int64 __fastcall sub_D1A870(_QWORD *a1)
{
  unsigned __int8 *v2; // r12
  unsigned __int8 v3; // al
  __int64 v4; // rbx
  int v5; // eax
  int v6; // edx
  __int64 v7; // rsi
  unsigned int v8; // eax
  unsigned __int8 **v9; // r14
  unsigned __int8 *v10; // rcx
  unsigned __int64 v11; // r15
  unsigned __int8 **v12; // rsi
  unsigned __int8 **v13; // rdx
  unsigned __int8 **v14; // rax
  __int64 v15; // rcx
  unsigned int v16; // edi
  unsigned __int64 v17; // rdx
  _QWORD *v18; // rax
  int v19; // eax
  __int64 v20; // rcx
  int v21; // esi
  unsigned int v22; // edx
  unsigned __int8 **v23; // rax
  unsigned __int8 *v24; // rdi
  __int64 v25; // rax
  _QWORD *v26; // r12
  __int64 v27; // rax
  __int64 *v29; // rax
  unsigned __int8 **v30; // rsi
  unsigned __int8 **v31; // rdx
  unsigned __int8 **v32; // rax
  __int64 v33; // rcx
  _QWORD *v34; // rdx
  _QWORD *v35; // rcx
  unsigned __int64 v36; // r9
  int v37; // r8d
  unsigned int v38; // r10d
  unsigned __int8 **v39; // rsi
  unsigned __int8 *v40; // r11
  unsigned int v41; // esi
  int v42; // r8d
  int v43; // eax
  int v44; // r8d
  __int64 *v45; // rax
  int v46; // edi
  _QWORD *v47; // rdx
  _QWORD *v48; // rcx
  _QWORD *v49; // rax
  int v50; // esi
  int v51; // ebx

  v2 = (unsigned __int8 *)a1[3];
  v3 = *v2;
  if ( !*v2 )
  {
    v4 = a1[4];
    v5 = *(_DWORD *)(v4 + 296);
    if ( !v5 )
      goto LABEL_7;
    v6 = v5 - 1;
    v7 = *(_QWORD *)(v4 + 280);
    v8 = (v5 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v9 = (unsigned __int8 **)(v7 + 16LL * v8);
    v10 = *v9;
    if ( v2 != *v9 )
    {
      v46 = 1;
      while ( v10 != (unsigned __int8 *)-4096LL )
      {
        v8 = v6 & (v46 + v8);
        v9 = (unsigned __int8 **)(v7 + 16LL * v8);
        v10 = *v9;
        if ( v2 == *v9 )
          goto LABEL_4;
        ++v46;
      }
      goto LABEL_7;
    }
LABEL_4:
    v11 = (unsigned __int64)v9[1] & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11 )
    {
      if ( (*(_BYTE *)(v11 + 8) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v11 + 16), 16LL * *(unsigned int *)(v11 + 24), 8);
      j_j___libc_free_0(v11, 272);
    }
    *v9 = (unsigned __int8 *)-8192LL;
    --*(_DWORD *)(v4 + 288);
    ++*(_DWORD *)(v4 + 292);
    v3 = *v2;
  }
  v4 = a1[4];
  if ( v3 > 3u )
    goto LABEL_20;
LABEL_7:
  if ( *(_BYTE *)(v4 + 68) )
  {
    v12 = *(unsigned __int8 ***)(v4 + 48);
    v13 = &v12[*(unsigned int *)(v4 + 60)];
    if ( v12 == v13 )
      goto LABEL_20;
    v14 = *(unsigned __int8 ***)(v4 + 48);
    while ( v2 != *v14 )
    {
      if ( v13 == ++v14 )
        goto LABEL_20;
    }
    v15 = (unsigned int)(*(_DWORD *)(v4 + 60) - 1);
    *(_DWORD *)(v4 + 60) = v15;
    *v14 = v12[v15];
    ++*(_QWORD *)(v4 + 40);
  }
  else
  {
    v29 = sub_C8CA60(v4 + 40, (__int64)v2);
    if ( !v29 )
    {
LABEL_19:
      v4 = a1[4];
      goto LABEL_20;
    }
    *v29 = -2;
    ++*(_DWORD *)(v4 + 64);
    ++*(_QWORD *)(v4 + 40);
  }
  v4 = a1[4];
  if ( *(_BYTE *)(v4 + 172) )
  {
    v30 = *(unsigned __int8 ***)(v4 + 152);
    v31 = &v30[*(unsigned int *)(v4 + 164)];
    if ( v30 == v31 )
      goto LABEL_41;
    v32 = *(unsigned __int8 ***)(v4 + 152);
    while ( v2 != *v32 )
    {
      if ( v31 == ++v32 )
        goto LABEL_41;
    }
    v33 = (unsigned int)(*(_DWORD *)(v4 + 164) - 1);
    *(_DWORD *)(v4 + 164) = v33;
    *v32 = v30[v33];
    ++*(_QWORD *)(v4 + 144);
  }
  else
  {
    v45 = sub_C8CA60(v4 + 144, (__int64)v2);
    if ( !v45 )
    {
      v4 = a1[4];
      goto LABEL_41;
    }
    *v45 = -2;
    ++*(_DWORD *)(v4 + 168);
    ++*(_QWORD *)(v4 + 144);
  }
  v4 = a1[4];
  if ( *(_DWORD *)(v4 + 256) )
  {
    v47 = *(_QWORD **)(v4 + 248);
    v48 = &v47[2 * *(unsigned int *)(v4 + 264)];
    if ( v47 != v48 )
    {
      while ( 1 )
      {
        v49 = v47;
        if ( *v47 != -4096 && *v47 != -8192 )
          break;
        v47 += 2;
        if ( v48 == v47 )
          goto LABEL_41;
      }
      if ( v48 != v47 )
      {
        if ( (unsigned __int8 *)v47[1] == v2 )
          goto LABEL_83;
        while ( 1 )
        {
          v49 += 2;
          if ( v49 == v48 )
            break;
          while ( *v49 == -4096 || *v49 == -8192 )
          {
            v49 += 2;
            if ( v48 == v49 )
              goto LABEL_41;
          }
          if ( v49 == v48 )
            break;
          if ( (unsigned __int8 *)v49[1] == v2 )
          {
LABEL_83:
            *v49 = -8192;
            --*(_DWORD *)(v4 + 256);
            ++*(_DWORD *)(v4 + 260);
            v4 = a1[4];
          }
        }
      }
    }
  }
LABEL_41:
  if ( *(_DWORD *)(v4 + 288) )
  {
    v34 = *(_QWORD **)(v4 + 280);
    v35 = &v34[2 * *(unsigned int *)(v4 + 296)];
    if ( v34 != v35 )
    {
      while ( 1 )
      {
        v18 = v34;
        if ( *v34 != -4096 && *v34 != -8192 )
          break;
        v34 += 2;
        if ( v35 == v34 )
          goto LABEL_20;
      }
      if ( v34 != v35 )
      {
        v16 = ((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4);
        v17 = v34[1] & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v18[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          goto LABEL_49;
        while ( 1 )
        {
          while ( 1 )
          {
            do
            {
              v18 += 2;
              if ( v18 == v35 )
                goto LABEL_19;
              while ( *v18 == -8192 || *v18 == -4096 )
              {
                v18 += 2;
                if ( v35 == v18 )
                  goto LABEL_19;
              }
              if ( v18 == v35 )
                goto LABEL_19;
              v17 = v18[1] & 0xFFFFFFFFFFFFFFF8LL;
            }
            while ( !v17 );
LABEL_49:
            if ( (*(_BYTE *)(v17 + 8) & 1) == 0 )
              break;
            v36 = v17 + 16;
            v37 = 15;
LABEL_51:
            v38 = v16 & v37;
            v39 = (unsigned __int8 **)(v36 + 16LL * (v16 & v37));
            v40 = *v39;
            if ( v2 == *v39 )
            {
LABEL_52:
              *v39 = (unsigned __int8 *)-8192LL;
              v41 = *(_DWORD *)(v17 + 8);
              ++*(_DWORD *)(v17 + 12);
              *(_DWORD *)(v17 + 8) = (2 * (v41 >> 1) - 2) | v41 & 1;
            }
            else
            {
              v50 = 1;
              while ( v40 != (unsigned __int8 *)-4096LL )
              {
                v51 = v50 + 1;
                v38 = v37 & (v50 + v38);
                v39 = (unsigned __int8 **)(v36 + 16LL * v38);
                v40 = *v39;
                if ( v2 == *v39 )
                  goto LABEL_52;
                v50 = v51;
              }
            }
          }
          v42 = *(_DWORD *)(v17 + 24);
          v36 = *(_QWORD *)(v17 + 16);
          if ( v42 )
          {
            v37 = v42 - 1;
            goto LABEL_51;
          }
        }
      }
    }
  }
LABEL_20:
  v19 = *(_DWORD *)(v4 + 264);
  v20 = *(_QWORD *)(v4 + 248);
  if ( v19 )
  {
    v21 = v19 - 1;
    v22 = (v19 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v23 = (unsigned __int8 **)(v20 + 16LL * v22);
    v24 = *v23;
    if ( v2 == *v23 )
    {
LABEL_22:
      *v23 = (unsigned __int8 *)-8192LL;
      --*(_DWORD *)(v4 + 256);
      ++*(_DWORD *)(v4 + 260);
    }
    else
    {
      v43 = 1;
      while ( v24 != (unsigned __int8 *)-4096LL )
      {
        v44 = v43 + 1;
        v22 = v21 & (v43 + v22);
        v23 = (unsigned __int8 **)(v20 + 16LL * v22);
        v24 = *v23;
        if ( v2 == *v23 )
          goto LABEL_22;
        v43 = v44;
      }
    }
  }
  v25 = a1[3];
  if ( v25 )
  {
    if ( v25 != -4096 && v25 != -8192 )
      sub_BD60C0(a1 + 1);
    a1[3] = 0;
  }
  v26 = (_QWORD *)a1[5];
  --*(_QWORD *)(a1[4] + 352LL);
  sub_2208CA0(v26);
  v26[2] = &unk_49DB368;
  v27 = v26[5];
  if ( v27 != 0 && v27 != -4096 && v27 != -8192 )
    sub_BD60C0(v26 + 3);
  return j_j___libc_free_0(v26, 64);
}
