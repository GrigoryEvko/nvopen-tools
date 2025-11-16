// Function: sub_D9EE30
// Address: 0xd9ee30
//
void __fastcall sub_D9EE30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 *a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  __int64 v11; // r15
  unsigned __int64 v12; // rsi
  __int64 v13; // r15
  __int64 v14; // rbx
  unsigned __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // r13
  __int64 v21; // rbx
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rbx
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rbx
  char **v36; // r15
  unsigned __int64 v37; // r14
  __int64 v38; // rsi
  __int64 v39; // rax
  char v40; // al
  char **v41; // rsi
  __int64 v42; // rdi
  __int64 v43; // r15
  __int64 v44; // rdi
  __int64 v45; // rax
  unsigned __int64 v46; // r14
  __int64 v47; // r15
  __int64 v48; // rsi
  __int64 v49; // rax
  char v50; // al
  __int64 v51; // rdi
  __int64 v52; // [rsp-58h] [rbp-58h]
  __int64 v53; // [rsp-58h] [rbp-58h]
  unsigned __int64 v54; // [rsp-50h] [rbp-50h]
  int v55; // [rsp-44h] [rbp-44h]
  unsigned __int64 v56; // [rsp-40h] [rbp-40h]
  unsigned __int64 v57; // [rsp-40h] [rbp-40h]
  __int64 v58; // [rsp-40h] [rbp-40h]

  if ( a1 == a2 )
    return;
  v6 = a2 + 16;
  v9 = *(_QWORD *)a1;
  v10 = *(unsigned int *)(a1 + 8);
  v11 = *(_QWORD *)a1;
  if ( *(_QWORD *)a2 != a2 + 16 )
  {
    v24 = v9 + 112 * v10;
    if ( v9 != v24 )
    {
      do
      {
        v24 -= 112;
        v25 = *(_QWORD *)(v24 + 64);
        if ( v25 != v24 + 80 )
          _libc_free(v25, a2);
        if ( *(_BYTE *)(v24 + 32) )
          *(_QWORD *)(v24 + 24) = 0;
        *(_QWORD *)v24 = &unk_49DB368;
        v26 = *(_QWORD *)(v24 + 24);
        if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
          sub_BD60C0((_QWORD *)(v24 + 8));
      }
      while ( v9 != v24 );
      v24 = *(_QWORD *)a1;
    }
    if ( v24 != a1 + 16 )
      _libc_free(v24, a2);
    *(_QWORD *)a1 = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
    *(_QWORD *)a2 = v6;
    *(_QWORD *)(a2 + 8) = 0;
    return;
  }
  v12 = *(unsigned int *)(a2 + 8);
  v55 = v12;
  if ( v12 > v10 )
  {
    if ( v12 > *(unsigned int *)(a1 + 12) )
    {
      v43 = v9 + 112 * v10;
      while ( v9 != v43 )
      {
        v43 -= 112;
        v44 = *(_QWORD *)(v43 + 64);
        if ( v44 != v43 + 80 )
          _libc_free(v44, v12);
        if ( *(_BYTE *)(v43 + 32) )
          *(_QWORD *)(v43 + 24) = 0;
        v45 = *(_QWORD *)(v43 + 24);
        *(_QWORD *)v43 = &unk_49DB368;
        LOBYTE(a4) = v45 != 0;
        LOBYTE(v10) = v45 != -4096;
        if ( ((unsigned __int8)v10 & (v45 != 0)) != 0 && v45 != -8192 )
          sub_BD60C0((_QWORD *)(v43 + 8));
      }
      *(_DWORD *)(a1 + 8) = 0;
      sub_D9E710(a1, v12, v10, a4, (__int64)a5, a6);
      v6 = *(_QWORD *)a2;
      v12 = *(unsigned int *)(a2 + 8);
      v10 = 0;
      v9 = *(_QWORD *)a1;
      v13 = *(_QWORD *)a2;
LABEL_6:
      v14 = v10 + v9;
      v15 = 112 * v12 + v6;
      while ( v15 != v13 )
      {
        while ( 1 )
        {
          if ( v14 )
          {
            v16 = *(_QWORD *)(v13 + 8);
            *(_QWORD *)(v14 + 16) = 0;
            *(_QWORD *)(v14 + 8) = v16 & 6;
            v17 = *(_QWORD *)(v13 + 24);
            *(_QWORD *)(v14 + 24) = v17;
            LOBYTE(a4) = v17 != -4096;
            LOBYTE(v10) = v17 != 0;
            if ( ((v17 != 0) & (unsigned __int8)a4) != 0 && v17 != -8192 )
            {
              v12 = *(_QWORD *)(v13 + 8) & 0xFFFFFFFFFFFFFFF8LL;
              sub_BD6050((unsigned __int64 *)(v14 + 8), v12);
            }
            *(_QWORD *)v14 = &unk_49DE8C0;
            *(_BYTE *)(v14 + 32) = *(_BYTE *)(v13 + 32);
            *(_QWORD *)(v14 + 40) = *(_QWORD *)(v13 + 40);
            *(_QWORD *)(v14 + 48) = *(_QWORD *)(v13 + 48);
            v18 = *(_QWORD *)(v13 + 56);
            *(_DWORD *)(v14 + 72) = 0;
            *(_QWORD *)(v14 + 56) = v18;
            *(_QWORD *)(v14 + 64) = v14 + 80;
            *(_DWORD *)(v14 + 76) = 4;
            if ( *(_DWORD *)(v13 + 72) )
              break;
          }
          v13 += 112;
          v14 += 112;
          if ( v15 == v13 )
            goto LABEL_15;
        }
        v12 = v13 + 64;
        v19 = v14 + 64;
        v13 += 112;
        v14 += 112;
        sub_D91460(v19, (char **)v12, v10, a4, (__int64)a5, a6);
      }
LABEL_15:
      *(_DWORD *)(a1 + 8) = v55;
      v20 = *(_QWORD *)a2;
      v21 = *(_QWORD *)a2 + 112LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v21 )
      {
        do
        {
          v21 -= 112;
          v22 = *(_QWORD *)(v21 + 64);
          if ( v22 != v21 + 80 )
            _libc_free(v22, v12);
          if ( *(_BYTE *)(v21 + 32) )
            *(_QWORD *)(v21 + 24) = 0;
          v23 = *(_QWORD *)(v21 + 24);
          *(_QWORD *)v21 = &unk_49DB368;
          if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
            sub_BD60C0((_QWORD *)(v21 + 8));
        }
        while ( v20 != v21 );
      }
      goto LABEL_24;
    }
    v13 = v6;
    if ( !*(_DWORD *)(a1 + 8) )
      goto LABEL_6;
    v35 = v9 + 64;
    v52 = 112 * v10;
    v10 *= 112LL;
    v36 = (char **)(a2 + 80);
    v37 = v35 + v10;
    while ( 1 )
    {
      if ( *(_BYTE *)(v35 - 32) )
      {
        *(_QWORD *)(v35 - 40) = 0;
        v39 = (__int64)*(v36 - 5);
        a5 = (unsigned __int64 *)(v35 - 56);
        if ( v39 )
          goto LABEL_63;
      }
      else
      {
        v38 = *(_QWORD *)(v35 - 40);
        v39 = (__int64)*(v36 - 5);
        if ( v39 != v38 )
        {
          a5 = (unsigned __int64 *)(v35 - 56);
          LOBYTE(a6) = v38 != 0;
          if ( v38 != -4096 && v38 != 0 && v38 != -8192 )
          {
            v54 = v10;
            sub_BD60C0((_QWORD *)(v35 - 56));
            v39 = (__int64)*(v36 - 5);
            v10 = v54;
            a5 = (unsigned __int64 *)(v35 - 56);
          }
LABEL_63:
          *(_QWORD *)(v35 - 40) = v39;
          if ( v39 != -4096 && v39 != 0 && v39 != -8192 )
          {
            v56 = v10;
            sub_BD6050(a5, (unsigned __int64)*(v36 - 7) & 0xFFFFFFFFFFFFFFF8LL);
            v10 = v56;
          }
        }
      }
      v40 = *((_BYTE *)v36 - 32);
      v41 = v36;
      v42 = v35;
      v35 += 112;
      v57 = v10;
      v36 += 14;
      *(_BYTE *)(v35 - 144) = v40;
      *(_QWORD *)(v35 - 136) = *(v36 - 17);
      *(_QWORD *)(v35 - 128) = *(v36 - 16);
      *(_QWORD *)(v35 - 120) = *(v36 - 15);
      sub_D91460(v42, v41, v10, a4, (__int64)a5, a6);
      v10 = v57;
      if ( v35 == v37 )
      {
        v6 = *(_QWORD *)a2;
        v12 = *(unsigned int *)(a2 + 8);
        v9 = *(_QWORD *)a1;
        v13 = *(_QWORD *)a2 + v52;
        goto LABEL_6;
      }
    }
  }
  v27 = *(_QWORD *)a1;
  if ( !v12 )
    goto LABEL_39;
  v46 = a2 + 80;
  v53 = 112 * v12;
  v47 = v9 + 64;
  v58 = v9 + 64 + 112 * v12;
  do
  {
    if ( *(_BYTE *)(v47 - 32) )
    {
      *(_QWORD *)(v47 - 40) = 0;
      v49 = *(_QWORD *)(v46 - 40);
      a5 = (unsigned __int64 *)(v47 - 56);
      if ( !v49 )
        goto LABEL_90;
    }
    else
    {
      v48 = *(_QWORD *)(v47 - 40);
      v49 = *(_QWORD *)(v46 - 40);
      if ( v49 == v48 )
        goto LABEL_90;
      a5 = (unsigned __int64 *)(v47 - 56);
      LOBYTE(a6) = v48 != -4096;
      if ( ((v48 != 0) & (unsigned __int8)a6) != 0 && v48 != -8192 )
      {
        sub_BD60C0((_QWORD *)(v47 - 56));
        v49 = *(_QWORD *)(v46 - 40);
        a5 = (unsigned __int64 *)(v47 - 56);
      }
    }
    *(_QWORD *)(v47 - 40) = v49;
    if ( v49 != -4096 && v49 != 0 && v49 != -8192 )
      sub_BD6050(a5, *(_QWORD *)(v46 - 56) & 0xFFFFFFFFFFFFFFF8LL);
LABEL_90:
    v50 = *(_BYTE *)(v46 - 32);
    v12 = v46;
    v51 = v47;
    v46 += 112LL;
    v47 += 112;
    *(_BYTE *)(v47 - 144) = v50;
    *(_QWORD *)(v47 - 136) = *(_QWORD *)(v46 - 136);
    *(_QWORD *)(v47 - 128) = *(_QWORD *)(v46 - 128);
    *(_QWORD *)(v47 - 120) = *(_QWORD *)(v46 - 120);
    sub_D91460(v51, (char **)v12, v10, a4, (__int64)a5, a6);
  }
  while ( v47 != v58 );
  v27 = *(_QWORD *)a1;
  v10 = *(unsigned int *)(a1 + 8);
  v11 = v9 + v53;
LABEL_39:
  v28 = v27 + 112 * v10;
  while ( v11 != v28 )
  {
    v28 -= 112;
    v29 = *(_QWORD *)(v28 + 64);
    if ( v29 != v28 + 80 )
      _libc_free(v29, v12);
    if ( *(_BYTE *)(v28 + 32) )
      *(_QWORD *)(v28 + 24) = 0;
    v30 = *(_QWORD *)(v28 + 24);
    *(_QWORD *)v28 = &unk_49DB368;
    if ( v30 != -4096 && v30 != 0 && v30 != -8192 )
      sub_BD60C0((_QWORD *)(v28 + 8));
  }
  *(_DWORD *)(a1 + 8) = v55;
  v31 = *(_QWORD *)a2;
  v32 = *(_QWORD *)a2 + 112LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v32 )
  {
    do
    {
      v32 -= 112;
      v33 = *(_QWORD *)(v32 + 64);
      if ( v33 != v32 + 80 )
        _libc_free(v33, v12);
      if ( *(_BYTE *)(v32 + 32) )
        *(_QWORD *)(v32 + 24) = 0;
      v34 = *(_QWORD *)(v32 + 24);
      *(_QWORD *)v32 = &unk_49DB368;
      if ( v34 != 0 && v34 != -4096 && v34 != -8192 )
        sub_BD60C0((_QWORD *)(v32 + 8));
    }
    while ( v31 != v32 );
  }
LABEL_24:
  *(_DWORD *)(a2 + 8) = 0;
}
