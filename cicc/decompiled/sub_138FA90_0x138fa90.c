// Function: sub_138FA90
// Address: 0x138fa90
//
__int64 __fastcall sub_138FA90(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // r8
  unsigned int *v5; // rcx
  unsigned int v6; // edx
  unsigned int *v7; // rax
  unsigned int v8; // eax
  unsigned int *v9; // rsi
  unsigned int v10; // esi
  __int64 v11; // r9
  unsigned int v12; // edx
  __int64 v13; // rsi
  unsigned int v14; // ecx
  int *v15; // rsi
  int v16; // ecx
  __int64 v17; // rdx
  __int64 v18; // r9
  __int64 v19; // r8
  unsigned int *v20; // rdx
  unsigned int v21; // ecx
  unsigned int v22; // eax
  unsigned int *v23; // r9
  unsigned int v24; // r9d
  __int64 v25; // r9
  unsigned int v26; // edx
  unsigned int v27; // ecx
  int *v28; // rsi
  int v29; // ecx
  __int64 v30; // rdx
  __int64 v31; // r9
  __int64 v32; // r8
  unsigned int v33; // edx
  __int64 v34; // rcx
  unsigned int v35; // ecx
  int *v36; // r10
  int v37; // r10d
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // r8
  __int64 v41; // r9
  unsigned int v42; // edx
  unsigned int v43; // ecx
  int *v44; // r10
  int v45; // r10d
  __int64 v46; // rsi
  unsigned int *v47; // r8
  unsigned int v48; // edx
  unsigned int v49; // eax
  unsigned int *v50; // r9
  unsigned int v51; // r9d
  __int64 result; // rax
  __int64 v53; // rdx
  __int64 v54; // r9
  __int64 v55; // r8
  unsigned int v56; // edx
  __int64 v57; // rsi
  unsigned int v58; // esi
  int *v59; // r10
  int v60; // r10d

  v3 = *(_QWORD *)(a1 + 32);
  v5 = (unsigned int *)(v3 + 32LL * a2);
  v6 = v5[6];
  v7 = v5;
  if ( v6 != -1 )
  {
    v8 = v5[6];
    do
    {
      v9 = (unsigned int *)(v3 + 32LL * v8);
      v8 = v9[6];
    }
    while ( v8 != -1 );
    v10 = *v9;
    while ( 1 )
    {
      v5[6] = v10;
      v7 = (unsigned int *)(v3 + 32LL * v6);
      v3 = *(_QWORD *)(a1 + 32);
      v6 = v7[6];
      if ( v6 == -1 )
        break;
      v5 = v7;
    }
  }
  v11 = v3 + 32LL * a3;
  v12 = *(_DWORD *)(v11 + 24);
  v13 = v11;
  v14 = v12;
  if ( v12 != -1 )
  {
    do
    {
      v15 = (int *)(v3 + 32LL * v14);
      v14 = v15[6];
    }
    while ( v14 != -1 );
    v16 = *v15;
    while ( 1 )
    {
      *(_DWORD *)(v11 + 24) = v16;
      v13 = v3 + 32LL * v12;
      v12 = *(_DWORD *)(v13 + 24);
      if ( v12 == -1 )
        break;
      v3 = *(_QWORD *)(a1 + 32);
      v11 = v13;
    }
  }
LABEL_12:
  v17 = v7[2];
  if ( (_DWORD)v17 == -1 )
  {
LABEL_28:
    v30 = *(unsigned int *)(v13 + 8);
    if ( (_DWORD)v30 != -1 )
    {
      v7[2] = v30;
      v31 = *(_QWORD *)(a1 + 32);
      v32 = v31 + 32 * v30;
      v33 = *(_DWORD *)(v32 + 24);
      v34 = v32;
      if ( v33 != -1 )
      {
        v35 = *(_DWORD *)(v32 + 24);
        do
        {
          v36 = (int *)(v31 + 32LL * v35);
          v35 = v36[6];
        }
        while ( v35 != -1 );
        v37 = *v36;
        while ( 1 )
        {
          *(_DWORD *)(v32 + 24) = v37;
          v34 = v31 + 32LL * v33;
          v33 = *(_DWORD *)(v34 + 24);
          if ( v33 == -1 )
            break;
          v31 = *(_QWORD *)(a1 + 32);
          v32 = v34;
        }
      }
      *(_DWORD *)(v34 + 12) = *v7;
    }
  }
  else
  {
    while ( 1 )
    {
      v18 = *(unsigned int *)(v13 + 8);
      if ( (_DWORD)v18 == -1 )
        break;
      v19 = *(_QWORD *)(a1 + 32);
      v20 = (unsigned int *)(v19 + 32 * v17);
      v21 = v20[6];
      v7 = v20;
      if ( v21 != -1 )
      {
        v22 = v20[6];
        do
        {
          v23 = (unsigned int *)(v19 + 32LL * v22);
          v22 = v23[6];
        }
        while ( v22 != -1 );
        v24 = *v23;
        while ( 1 )
        {
          v20[6] = v24;
          v7 = (unsigned int *)(v19 + 32LL * v21);
          v21 = v7[6];
          if ( v21 == -1 )
            break;
          v19 = *(_QWORD *)(a1 + 32);
          v20 = v7;
        }
        v18 = *(unsigned int *)(v13 + 8);
        v19 = *(_QWORD *)(a1 + 32);
      }
      v25 = v19 + 32 * v18;
      v26 = *(_DWORD *)(v25 + 24);
      v13 = v25;
      if ( v26 == -1 )
        goto LABEL_12;
      v27 = *(_DWORD *)(v25 + 24);
      do
      {
        v28 = (int *)(v19 + 32LL * v27);
        v27 = v28[6];
      }
      while ( v27 != -1 );
      v29 = *v28;
      while ( 1 )
      {
        *(_DWORD *)(v25 + 24) = v29;
        v13 = v19 + 32LL * v26;
        v26 = *(_DWORD *)(v13 + 24);
        if ( v26 == -1 )
          break;
        v19 = *(_QWORD *)(a1 + 32);
        v25 = v13;
      }
      v17 = v7[2];
      if ( (_DWORD)v17 == -1 )
        goto LABEL_28;
    }
  }
  v38 = v13;
  if ( v7[3] != -1 )
  {
    v39 = *((_QWORD *)v7 + 2) | *(_QWORD *)(v13 + 16);
    if ( *(_DWORD *)(v13 + 12) == -1 )
    {
LABEL_52:
      *((_QWORD *)v7 + 2) |= *(_QWORD *)(v13 + 16);
      result = *v7;
      *(_DWORD *)(v13 + 24) = result;
      return result;
    }
    while ( 1 )
    {
      *((_QWORD *)v7 + 2) = v39;
      v40 = *(_QWORD *)(a1 + 32);
      v41 = v40 + 32LL * *(unsigned int *)(v13 + 12);
      v42 = *(_DWORD *)(v41 + 24);
      v38 = v41;
      if ( v42 != -1 )
      {
        v43 = *(_DWORD *)(v41 + 24);
        do
        {
          v44 = (int *)(v40 + 32LL * v43);
          v43 = v44[6];
        }
        while ( v43 != -1 );
        v45 = *v44;
        while ( 1 )
        {
          *(_DWORD *)(v41 + 24) = v45;
          v38 = v40 + 32LL * v42;
          v42 = *(_DWORD *)(v38 + 24);
          if ( v42 == -1 )
            break;
          v40 = *(_QWORD *)(a1 + 32);
          v41 = v38;
        }
      }
      *(_DWORD *)(v13 + 24) = *v7;
      v46 = *(_QWORD *)(a1 + 32);
      v47 = (unsigned int *)(v46 + 32LL * v7[3]);
      v48 = v47[6];
      v7 = v47;
      if ( v48 != -1 )
      {
        v49 = v47[6];
        do
        {
          v50 = (unsigned int *)(v46 + 32LL * v49);
          v49 = v50[6];
        }
        while ( v49 != -1 );
        v51 = *v50;
        while ( 1 )
        {
          v47[6] = v51;
          v7 = (unsigned int *)(v46 + 32LL * v48);
          v48 = v7[6];
          if ( v48 == -1 )
            break;
          v46 = *(_QWORD *)(a1 + 32);
          v47 = v7;
        }
      }
      if ( v7[3] == -1 )
        break;
      v13 = v38;
      v39 = *((_QWORD *)v7 + 2) | *(_QWORD *)(v38 + 16);
      if ( *(_DWORD *)(v38 + 12) == -1 )
        goto LABEL_52;
    }
  }
  v53 = *(unsigned int *)(v38 + 12);
  if ( (_DWORD)v53 != -1 )
  {
    v7[3] = v53;
    v54 = *(_QWORD *)(a1 + 32);
    v55 = v54 + 32 * v53;
    v56 = *(_DWORD *)(v55 + 24);
    v57 = v55;
    if ( v56 != -1 )
    {
      v58 = *(_DWORD *)(v55 + 24);
      do
      {
        v59 = (int *)(v54 + 32LL * v58);
        v58 = v59[6];
      }
      while ( v58 != -1 );
      v60 = *v59;
      while ( 1 )
      {
        *(_DWORD *)(v55 + 24) = v60;
        v57 = v54 + 32LL * v56;
        v56 = *(_DWORD *)(v57 + 24);
        if ( v56 == -1 )
          break;
        v54 = *(_QWORD *)(a1 + 32);
        v55 = v57;
      }
    }
    *(_DWORD *)(v57 + 8) = *v7;
  }
  *((_QWORD *)v7 + 2) |= *(_QWORD *)(v38 + 16);
  result = *v7;
  *(_DWORD *)(v38 + 24) = result;
  return result;
}
