// Function: sub_102BD40
// Address: 0x102bd40
//
__int64 __fastcall sub_102BD40(__int64 a1)
{
  __int64 v2; // rsi
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rsi
  _QWORD *v8; // r12
  __int64 v9; // rsi
  _QWORD *v10; // r13
  __int64 v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // r15
  __int64 v14; // rsi
  __int64 *v15; // r14
  __int64 *v16; // r12
  __int64 i; // rax
  __int64 v18; // rdi
  unsigned int v19; // ecx
  __int64 *v20; // r12
  __int64 *v21; // r13
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // r13
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // r13
  __int64 v33; // rdi
  __int64 v34; // rax
  _QWORD *v35; // r12
  _QWORD *v36; // r13
  __int64 v37; // rdi
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // r12
  __int64 v41; // r13
  __int64 v42; // rdi
  __int64 v43; // rax
  _QWORD *v44; // r12
  _QWORD *v45; // r13
  __int64 v46; // rdi
  __int64 v47; // rsi
  __int64 v48; // rax
  __int64 v49; // r12
  __int64 v50; // r13
  __int64 v51; // rdi
  __int64 v52; // rax
  _QWORD *v54; // r12
  _QWORD *v55; // r13
  __int64 v56; // rax
  _QWORD v57[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v58; // [rsp+10h] [rbp-60h]
  __int64 v59; // [rsp+20h] [rbp-50h]
  __int64 v60; // [rsp+28h] [rbp-48h]
  __int64 v61; // [rsp+30h] [rbp-40h]

  v2 = 16LL * *(unsigned int *)(a1 + 1008);
  sub_C7D6A0(*(_QWORD *)(a1 + 992), v2, 8);
  if ( !*(_BYTE *)(a1 + 900) )
    _libc_free(*(_QWORD *)(a1 + 880), v2);
  if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
  {
    v4 = a1 + 520;
    v6 = a1 + 872;
  }
  else
  {
    v3 = *(unsigned int *)(a1 + 528);
    v4 = *(_QWORD *)(a1 + 520);
    v5 = 88 * v3;
    if ( !(_DWORD)v3 )
      goto LABEL_88;
    v6 = v4 + v5;
    if ( v4 + v5 == v4 )
      goto LABEL_88;
  }
  do
  {
    if ( *(_QWORD *)v4 != -8192 && *(_QWORD *)v4 != -4096 && (*(_BYTE *)(v4 + 16) & 1) == 0 )
      sub_C7D6A0(*(_QWORD *)(v4 + 24), 16LL * *(unsigned int *)(v4 + 32), 8);
    v4 += 88;
  }
  while ( v6 != v4 );
  if ( (*(_BYTE *)(a1 + 512) & 1) == 0 )
  {
    v4 = *(_QWORD *)(a1 + 520);
    v5 = 88LL * *(unsigned int *)(a1 + 528);
LABEL_88:
    sub_C7D6A0(v4, v5, 8);
  }
  v7 = *(unsigned int *)(a1 + 496);
  *(_QWORD *)(a1 + 416) = &unk_49DDC10;
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD **)(a1 + 480);
    v9 = 2 * v7;
    v10 = &v8[v9];
    do
    {
      if ( *v8 != -4096 && *v8 != -8192 )
      {
        v11 = v8[1];
        if ( v11 )
        {
          if ( (v11 & 4) != 0 )
          {
            v12 = (_QWORD *)(v11 & 0xFFFFFFFFFFFFFFF8LL);
            v13 = v12;
            if ( v12 )
            {
              if ( (_QWORD *)*v12 != v12 + 2 )
                _libc_free(*v12, v9 * 8);
              v9 = 6;
              j_j___libc_free_0(v13, 48);
            }
          }
        }
      }
      v8 += 2;
    }
    while ( v10 != v8 );
    v7 = *(unsigned int *)(a1 + 496);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 480), 16 * v7, 8);
  v14 = 16LL * *(unsigned int *)(a1 + 464);
  sub_C7D6A0(*(_QWORD *)(a1 + 448), v14, 8);
  nullsub_184();
  v15 = *(__int64 **)(a1 + 336);
  v16 = &v15[*(unsigned int *)(a1 + 344)];
  if ( v15 != v16 )
  {
    for ( i = *(_QWORD *)(a1 + 336); ; i = *(_QWORD *)(a1 + 336) )
    {
      v18 = *v15;
      v19 = (unsigned int)(((__int64)v15 - i) >> 3) >> 7;
      v14 = 4096LL << v19;
      if ( v19 >= 0x1E )
        v14 = 0x40000000000LL;
      ++v15;
      sub_C7D6A0(v18, v14, 16);
      if ( v16 == v15 )
        break;
    }
  }
  v20 = *(__int64 **)(a1 + 384);
  v21 = &v20[2 * *(unsigned int *)(a1 + 392)];
  if ( v20 != v21 )
  {
    do
    {
      v14 = v20[1];
      v22 = *v20;
      v20 += 2;
      sub_C7D6A0(v22, v14, 16);
    }
    while ( v21 != v20 );
    v21 = *(__int64 **)(a1 + 384);
  }
  if ( v21 != (__int64 *)(a1 + 400) )
    _libc_free(v21, v14);
  v23 = *(_QWORD *)(a1 + 336);
  if ( v23 != a1 + 352 )
    _libc_free(v23, v14);
  v24 = 24LL * *(unsigned int *)(a1 + 312);
  sub_C7D6A0(*(_QWORD *)(a1 + 296), v24, 8);
  v25 = *(unsigned int *)(a1 + 248);
  if ( (_DWORD)v25 )
  {
    v26 = *(_QWORD *)(a1 + 232);
    v27 = v26 + 72 * v25;
    do
    {
      while ( *(_QWORD *)v26 == -4096 || *(_QWORD *)v26 == -8192 || *(_BYTE *)(v26 + 36) )
      {
        v26 += 72;
        if ( v27 == v26 )
          goto LABEL_44;
      }
      v28 = *(_QWORD *)(v26 + 16);
      v26 += 72;
      _libc_free(v28, v24);
    }
    while ( v27 != v26 );
LABEL_44:
    v25 = *(unsigned int *)(a1 + 248);
  }
  v29 = 72 * v25;
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 72 * v25, 8);
  v30 = *(unsigned int *)(a1 + 216);
  if ( (_DWORD)v30 )
  {
    v31 = *(_QWORD *)(a1 + 200);
    v32 = v31 + 72 * v30;
    do
    {
      while ( *(_QWORD *)v31 == -8192 || *(_QWORD *)v31 == -4096 || *(_BYTE *)(v31 + 36) )
      {
        v31 += 72;
        if ( v32 == v31 )
          goto LABEL_52;
      }
      v33 = *(_QWORD *)(v31 + 16);
      v31 += 72;
      _libc_free(v33, v29);
    }
    while ( v32 != v31 );
LABEL_52:
    v30 = *(unsigned int *)(a1 + 216);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 200), 72 * v30, 8);
  v34 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v34 )
  {
    v35 = *(_QWORD **)(a1 + 168);
    v36 = &v35[5 * v34];
    do
    {
      if ( *v35 != -8192 && *v35 != -4096 )
      {
        v37 = v35[1];
        if ( v37 )
          j_j___libc_free_0(v37, v35[3] - v37);
      }
      v35 += 5;
    }
    while ( v36 != v35 );
    v34 = *(unsigned int *)(a1 + 184);
  }
  v38 = 40 * v34;
  sub_C7D6A0(*(_QWORD *)(a1 + 168), 40 * v34, 8);
  v39 = *(unsigned int *)(a1 + 152);
  if ( (_DWORD)v39 )
  {
    v40 = *(_QWORD *)(a1 + 136);
    v41 = v40 + 72 * v39;
    do
    {
      while ( *(_QWORD *)v40 == -8192 || *(_QWORD *)v40 == -4096 || *(_BYTE *)(v40 + 36) )
      {
        v40 += 72;
        if ( v41 == v40 )
          goto LABEL_68;
      }
      v42 = *(_QWORD *)(v40 + 16);
      v40 += 72;
      _libc_free(v42, v38);
    }
    while ( v41 != v40 );
LABEL_68:
    v39 = *(unsigned int *)(a1 + 152);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 136), 72 * v39, 8);
  v43 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v43 )
  {
    v44 = *(_QWORD **)(a1 + 104);
    v45 = &v44[10 * v43];
    do
    {
      if ( *v44 != -16 && *v44 != -4 )
      {
        v46 = v44[2];
        if ( v46 )
          j_j___libc_free_0(v46, v44[4] - v46);
      }
      v44 += 10;
    }
    while ( v45 != v44 );
    v43 = *(unsigned int *)(a1 + 120);
  }
  v47 = 80 * v43;
  sub_C7D6A0(*(_QWORD *)(a1 + 104), 80 * v43, 8);
  v48 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v48 )
  {
    v49 = *(_QWORD *)(a1 + 72);
    v50 = v49 + 72 * v48;
    do
    {
      while ( *(_QWORD *)v49 == -8192 || *(_QWORD *)v49 == -4096 || *(_BYTE *)(v49 + 36) )
      {
        v49 += 72;
        if ( v50 == v49 )
          goto LABEL_84;
      }
      v51 = *(_QWORD *)(v49 + 16);
      v49 += 72;
      _libc_free(v51, v47);
    }
    while ( v50 != v49 );
LABEL_84:
    v48 = *(unsigned int *)(a1 + 88);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 72 * v48, 8);
  v52 = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)v52 )
  {
    v54 = *(_QWORD **)(a1 + 40);
    v57[0] = 0;
    v57[1] = 0;
    v58 = -4096;
    v55 = &v54[6 * v52];
    v59 = 0;
    v60 = 0;
    v61 = -8192;
    do
    {
      v56 = v54[2];
      if ( v56 != -4096 && v56 != 0 && v56 != -8192 )
        sub_BD60C0(v54);
      v54 += 6;
    }
    while ( v55 != v54 );
    if ( v58 != -4096 && v58 != 0 )
      sub_BD60C0(v57);
    v52 = *(unsigned int *)(a1 + 56);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 40), 48 * v52, 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * *(unsigned int *)(a1 + 24), 8);
}
