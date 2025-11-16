// Function: sub_DBA850
// Address: 0xdba850
//
__int64 __fastcall sub_DBA850(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 *v5; // rsi
  __int64 v6; // rcx
  int v7; // r10d
  unsigned int v8; // eax
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 v11; // rdx
  _BYTE *v12; // rbx
  __int64 v13; // r8
  _BYTE *v14; // r14
  _BYTE *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v19; // rbx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // r11
  unsigned int v25; // eax
  __int64 v26; // rbx
  __int64 v27; // rdx
  __int64 *v28; // rbx
  __int64 *v29; // r13
  __int64 *v30; // rdi
  __int64 v31; // rax
  int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // rcx
  unsigned __int64 v35; // r8
  _BYTE *v36; // r14
  __int64 v37; // rax
  _BYTE *v38; // rbx
  _BYTE *v39; // rdi
  __int64 v40; // rax
  int v41; // eax
  __int64 v42; // rdi
  unsigned int v43; // eax
  int v44; // r10d
  int v45; // eax
  __int64 v46; // rdi
  int v47; // r10d
  unsigned int v48; // eax
  __int64 v49; // [rsp+8h] [rbp-198h]
  _BYTE *v50; // [rsp+10h] [rbp-190h]
  _BYTE *v51; // [rsp+18h] [rbp-188h]
  _BYTE *v52; // [rsp+20h] [rbp-180h]
  __int64 v53; // [rsp+28h] [rbp-178h]
  _BYTE v54[112]; // [rsp+30h] [rbp-170h] BYREF
  __int128 v55; // [rsp+A0h] [rbp-100h]
  __int128 v56; // [rsp+B0h] [rbp-F0h]
  __int64 *v57; // [rsp+C0h] [rbp-E0h] BYREF
  _BYTE *v58; // [rsp+C8h] [rbp-D8h] BYREF
  __int64 v59; // [rsp+D0h] [rbp-D0h] BYREF
  _BYTE v60[104]; // [rsp+D8h] [rbp-C8h] BYREF
  __int64 v61; // [rsp+140h] [rbp-60h]
  __int64 v62; // [rsp+148h] [rbp-58h]
  __int64 v63; // [rsp+150h] [rbp-50h]
  __int64 v64; // [rsp+158h] [rbp-48h]
  char v65; // [rsp+160h] [rbp-40h]

  v3 = a1 + 680;
  v49 = a2;
  v57 = (__int64 *)a2;
  v51 = v60;
  v58 = v60;
  v5 = (__int64 *)*(unsigned int *)(a1 + 704);
  v50 = v54;
  v52 = v54;
  v53 = 0x100000000LL;
  v59 = 0x100000000LL;
  v62 = 0;
  LOBYTE(v63) = 0;
  v64 = 0;
  v65 = 0;
  v55 = 0;
  v56 = 0;
  if ( !(_DWORD)v5 )
  {
    ++*(_QWORD *)(a1 + 680);
    goto LABEL_61;
  }
  v6 = *(_QWORD *)(a1 + 688);
  v7 = 1;
  v8 = ((_DWORD)v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = 0;
  v10 = v6 + 168LL * v8;
  v11 = *(_QWORD *)v10;
  if ( a2 == *(_QWORD *)v10 )
  {
LABEL_3:
    LOBYTE(v51) = 0;
    goto LABEL_4;
  }
  while ( v11 != -4096 )
  {
    if ( !v9 && v11 == -8192 )
      v9 = v10;
    v8 = ((_DWORD)v5 - 1) & (v7 + v8);
    v10 = v6 + 168LL * v8;
    v11 = *(_QWORD *)v10;
    if ( v49 == *(_QWORD *)v10 )
      goto LABEL_3;
    ++v7;
  }
  v32 = *(_DWORD *)(a1 + 696);
  if ( v9 )
    v10 = v9;
  ++*(_QWORD *)(a1 + 680);
  v33 = (unsigned int)(v32 + 1);
  if ( 4 * (int)v33 >= (unsigned int)(3 * (_DWORD)v5) )
  {
LABEL_61:
    sub_DB6980(v3, 2 * (_DWORD)v5);
    v41 = *(_DWORD *)(a1 + 704);
    if ( v41 )
    {
      v5 = v57;
      v35 = (unsigned int)(v41 - 1);
      v42 = *(_QWORD *)(a1 + 688);
      v43 = v35 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
      v10 = v42 + 168LL * v43;
      v33 = (unsigned int)(*(_DWORD *)(a1 + 696) + 1);
      v34 = *(_QWORD *)v10;
      if ( v57 == *(__int64 **)v10 )
        goto LABEL_42;
      v44 = 1;
      v9 = 0;
      while ( v34 != -4096 )
      {
        if ( v34 == -8192 && !v9 )
          v9 = v10;
        v43 = v35 & (v44 + v43);
        v10 = v42 + 168LL * v43;
        v34 = *(_QWORD *)v10;
        if ( v57 == *(__int64 **)v10 )
          goto LABEL_42;
        ++v44;
      }
LABEL_65:
      v34 = (__int64)v5;
      if ( v9 )
        v10 = v9;
      goto LABEL_42;
    }
LABEL_86:
    ++*(_DWORD *)(a1 + 696);
    BUG();
  }
  v34 = v49;
  v35 = (unsigned int)v5 >> 3;
  if ( (int)v5 - *(_DWORD *)(a1 + 700) - (int)v33 <= (unsigned int)v35 )
  {
    sub_DB6980(v3, (int)v5);
    v45 = *(_DWORD *)(a1 + 704);
    if ( v45 )
    {
      v5 = v57;
      v35 = (unsigned int)(v45 - 1);
      v9 = 0;
      v46 = *(_QWORD *)(a1 + 688);
      v47 = 1;
      v48 = v35 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
      v10 = v46 + 168LL * v48;
      v33 = (unsigned int)(*(_DWORD *)(a1 + 696) + 1);
      v34 = *(_QWORD *)v10;
      if ( v57 == *(__int64 **)v10 )
        goto LABEL_42;
      while ( v34 != -4096 )
      {
        if ( v34 == -8192 && !v9 )
          v9 = v10;
        v48 = v35 & (v47 + v48);
        v10 = v46 + 168LL * v48;
        v34 = *(_QWORD *)v10;
        if ( v57 == *(__int64 **)v10 )
          goto LABEL_42;
        ++v47;
      }
      goto LABEL_65;
    }
    goto LABEL_86;
  }
LABEL_42:
  *(_DWORD *)(a1 + 696) = v33;
  if ( *(_QWORD *)v10 != -4096 )
    --*(_DWORD *)(a1 + 700);
  *(_QWORD *)v10 = v34;
  *(_QWORD *)(v10 + 8) = v10 + 24;
  *(_QWORD *)(v10 + 16) = 0x100000000LL;
  if ( (_DWORD)v59 )
  {
    v5 = (__int64 *)&v58;
    sub_D9EE30(v10 + 8, (__int64)&v58, v33, v34, (unsigned __int64 *)v35, v9);
    v37 = (unsigned int)v59;
    *(_QWORD *)(v10 + 136) = v62;
    *(_BYTE *)(v10 + 144) = v63;
    *(_QWORD *)(v10 + 152) = v64;
    *(_BYTE *)(v10 + 160) = v65;
    v38 = v58;
    v36 = &v58[112 * v37];
    if ( v58 != v36 )
    {
      do
      {
        v36 -= 112;
        v39 = (_BYTE *)*((_QWORD *)v36 + 8);
        if ( v39 != v36 + 80 )
          _libc_free(v39, &v58);
        if ( v36[32] )
          *((_QWORD *)v36 + 3) = 0;
        v40 = *((_QWORD *)v36 + 3);
        *(_QWORD *)v36 = &unk_49DB368;
        if ( v40 != -4096 && v40 != 0 && v40 != -8192 )
          sub_BD60C0((_QWORD *)v36 + 1);
      }
      while ( v38 != v36 );
      v36 = v58;
    }
  }
  else
  {
    *(_QWORD *)(v10 + 136) = v62;
    *(_BYTE *)(v10 + 144) = v63;
    *(_QWORD *)(v10 + 152) = v64;
    *(_BYTE *)(v10 + 160) = v65;
    v36 = v58;
  }
  if ( v36 != v51 )
    _libc_free(v36, v5);
  LOBYTE(v51) = 1;
LABEL_4:
  v12 = v52;
  v13 = 112LL * (unsigned int)v53;
  v14 = &v52[v13];
  if ( v52 != &v52[v13] )
  {
    do
    {
      v14 -= 112;
      v15 = (_BYTE *)*((_QWORD *)v14 + 8);
      if ( v15 != v14 + 80 )
        _libc_free(v15, v5);
      if ( v14[32] )
        *((_QWORD *)v14 + 3) = 0;
      v16 = *((_QWORD *)v14 + 3);
      *(_QWORD *)v14 = &unk_49DB368;
      if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
        sub_BD60C0((_QWORD *)v14 + 1);
    }
    while ( v12 != v14 );
    v14 = v52;
  }
  if ( v14 != v50 )
    _libc_free(v14, v5);
  v17 = v10 + 8;
  if ( (_BYTE)v51 )
  {
    v19 = v49;
    sub_DB9040((__int64)&v57, a1, v49, 1u);
    v22 = *(unsigned int *)(a1 + 704);
    v23 = *(_QWORD *)(a1 + 688);
    if ( (_DWORD)v22 )
    {
      v24 = v19;
      v25 = (v22 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v20 = 5LL * v25;
      v26 = v23 + 168LL * v25;
      v27 = *(_QWORD *)v26;
      if ( v24 == *(_QWORD *)v26 )
      {
LABEL_20:
        v17 = v26 + 8;
        sub_D9EE30(v26 + 8, (__int64)&v57, v27, v22, (unsigned __int64 *)v20, v21);
        *(_QWORD *)(v26 + 136) = v61;
        *(_BYTE *)(v26 + 144) = v62;
        *(_QWORD *)(v26 + 152) = v63;
        *(_BYTE *)(v26 + 160) = v64;
        v28 = v57;
        v29 = &v57[14 * (unsigned int)v58];
        if ( v57 != v29 )
        {
          do
          {
            v29 -= 14;
            v30 = (__int64 *)v29[8];
            if ( v30 != v29 + 10 )
              _libc_free(v30, &v57);
            if ( *((_BYTE *)v29 + 32) )
              v29[3] = 0;
            v31 = v29[3];
            *v29 = (__int64)&unk_49DB368;
            if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
              sub_BD60C0(v29 + 1);
          }
          while ( v28 != v29 );
          v29 = v57;
        }
        if ( v29 != &v59 )
          _libc_free(v29, &v57);
        return v17;
      }
      v20 = 1;
      while ( v27 != -4096 )
      {
        v21 = (unsigned int)(v20 + 1);
        v25 = (v22 - 1) & (v20 + v25);
        v20 = 5LL * v25;
        v26 = v23 + 168LL * v25;
        v27 = *(_QWORD *)v26;
        if ( v49 == *(_QWORD *)v26 )
          goto LABEL_20;
        v20 = (unsigned int)v21;
      }
    }
    v27 = 5LL * (unsigned int)v22;
    v26 = v23 + 168LL * (unsigned int)v22;
    goto LABEL_20;
  }
  return v17;
}
