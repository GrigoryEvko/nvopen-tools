// Function: sub_3544A40
// Address: 0x3544a40
//
__int64 __fastcall sub_3544A40(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  int v4; // eax
  __int64 v6; // rcx
  int v9; // esi
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r8
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rcx
  __int64 v15; // r13
  _QWORD *v16; // r11
  _QWORD *v17; // rax
  __int64 v18; // r9
  __int64 v19; // rdx
  int v20; // r10d
  int v21; // r9d
  unsigned int v22; // ebx
  _QWORD *v23; // r11
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r9
  int v27; // r10d
  int v28; // eax
  __int64 v29; // rsi
  char v30; // r11
  int v31; // edi
  unsigned int i; // edx
  __int64 v33; // rcx
  unsigned __int64 v34; // rax
  int v35; // edi
  __int64 v36; // rsi
  unsigned __int64 v37; // rdx
  int v38; // edi
  unsigned int v39; // ecx
  __int64 *v40; // rax
  __int64 v41; // r9
  unsigned __int64 v42; // rsi
  _QWORD *v44; // rax
  _QWORD *v45; // rdi
  __int64 v46; // rcx
  __int64 v47; // rdx
  int v48; // r9d
  int v49; // eax
  int v50; // eax
  int v51; // eax
  int v52; // r10d
  int v53; // r9d
  int v54; // [rsp+Ch] [rbp-34h]

  LOBYTE(v3) = *(_WORD *)(a3 + 68) == 68 || *(_WORD *)(a3 + 68) == 0;
  if ( !(_BYTE)v3 )
    return v3;
  v4 = *(_DWORD *)(a2 + 960);
  v6 = *(_QWORD *)(a2 + 944);
  if ( v4 )
  {
    v9 = v4 - 1;
    v10 = (v4 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v11 = (__int64 *)(v6 + 16LL * v10);
    v12 = *v11;
    if ( a3 == *v11 )
    {
LABEL_4:
      v13 = v11[1];
      goto LABEL_5;
    }
    v50 = 1;
    while ( v12 != -4096 )
    {
      v53 = v50 + 1;
      v10 = v9 & (v50 + v10);
      v11 = (__int64 *)(v6 + 16LL * v10);
      v12 = *v11;
      if ( a3 == *v11 )
        goto LABEL_4;
      v50 = v53;
    }
  }
  v13 = 0;
LABEL_5:
  v14 = *(_QWORD **)(a1 + 48);
  v15 = a1 + 40;
  if ( v14 )
  {
    v16 = (_QWORD *)(a1 + 40);
    v17 = *(_QWORD **)(a1 + 48);
    do
    {
      while ( 1 )
      {
        v18 = v17[2];
        v19 = v17[3];
        if ( v17[4] >= v13 )
          break;
        v17 = (_QWORD *)v17[3];
        if ( !v19 )
          goto LABEL_10;
      }
      v16 = v17;
      v17 = (_QWORD *)v17[2];
    }
    while ( v18 );
LABEL_10:
    if ( (_QWORD *)v15 == v16 )
    {
      v20 = *(_DWORD *)(a1 + 80);
      v21 = *(_DWORD *)(a1 + 88);
      v22 = 0 % v21;
    }
    else
    {
      v20 = *(_DWORD *)(a1 + 80);
      if ( v16[4] > v13 )
        v16 = (_QWORD *)(a1 + 40);
      v21 = *(_DWORD *)(a1 + 88);
      v22 = (*((_DWORD *)v16 + 10) - v20) % v21;
    }
    v23 = (_QWORD *)(a1 + 40);
    do
    {
      while ( 1 )
      {
        v24 = v14[2];
        v25 = v14[3];
        if ( v14[4] >= v13 )
          break;
        v14 = (_QWORD *)v14[3];
        if ( !v25 )
          goto LABEL_18;
      }
      v23 = v14;
      v14 = (_QWORD *)v14[2];
    }
    while ( v24 );
LABEL_18:
    if ( (_QWORD *)v15 != v23 && v23[4] <= v13 )
    {
      v54 = (*((_DWORD *)v23 + 10) - v20) / v21;
      goto LABEL_21;
    }
  }
  else
  {
    v22 = 0;
  }
  v54 = -1;
LABEL_21:
  v26 = *(_QWORD *)(a3 + 24);
  v27 = 0;
  v28 = *(_DWORD *)(a3 + 40) & 0xFFFFFF;
  if ( v28 != 1 )
  {
    v29 = *(_QWORD *)(a3 + 32);
    v30 = 0;
    v31 = 0;
    for ( i = 1; i != v28; i += 2 )
    {
      while ( v26 != *(_QWORD *)(v29 + 40LL * (i + 1) + 24) )
      {
        i += 2;
        if ( v28 == i )
          goto LABEL_26;
      }
      v33 = i;
      v30 = v3;
      v31 = *(_DWORD *)(v29 + 40 * v33 + 8);
    }
LABEL_26:
    if ( v30 )
      v27 = v31;
  }
  v34 = sub_2EBEE10(*(_QWORD *)(a1 + 104), v27);
  v35 = *(_DWORD *)(a2 + 960);
  v36 = *(_QWORD *)(a2 + 944);
  v37 = v34;
  if ( v35 )
  {
    v38 = v35 - 1;
    v39 = v38 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
    v40 = (__int64 *)(v36 + 16LL * v39);
    v41 = *v40;
    if ( v37 == *v40 )
    {
LABEL_30:
      v42 = v40[1];
      if ( v42 )
      {
        LOBYTE(v3) = *(_WORD *)(*(_QWORD *)v42 + 68LL) == 68 || *(_WORD *)(*(_QWORD *)v42 + 68LL) == 0;
        if ( !(_BYTE)v3 )
        {
          v44 = *(_QWORD **)(a1 + 48);
          if ( v44 )
          {
            v45 = (_QWORD *)(a1 + 40);
            do
            {
              while ( 1 )
              {
                v46 = v44[2];
                v47 = v44[3];
                if ( v44[4] >= v42 )
                  break;
                v44 = (_QWORD *)v44[3];
                if ( !v47 )
                  goto LABEL_39;
              }
              v45 = v44;
              v44 = (_QWORD *)v44[2];
            }
            while ( v46 );
LABEL_39:
            if ( (_QWORD *)v15 != v45 && v45[4] <= v42 )
              v15 = (__int64)v45;
          }
          v48 = sub_3542500(a1, v42);
          v49 = (*(_DWORD *)(v15 + 40) - *(_DWORD *)(a1 + 80)) / *(_DWORD *)(a1 + 88);
          LOBYTE(v3) = v22 < (*(_DWORD *)(v15 + 40) - *(_DWORD *)(a1 + 80)) % *(_DWORD *)(a1 + 88);
          LOBYTE(v49) = v48 <= v54;
          v3 |= v49;
        }
      }
    }
    else
    {
      v51 = 1;
      while ( v41 != -4096 )
      {
        v52 = v51 + 1;
        v39 = v38 & (v51 + v39);
        v40 = (__int64 *)(v36 + 16LL * v39);
        v41 = *v40;
        if ( v37 == *v40 )
          goto LABEL_30;
        v51 = v52;
      }
    }
  }
  return v3;
}
