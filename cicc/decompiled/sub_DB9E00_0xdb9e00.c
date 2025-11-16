// Function: sub_DB9E00
// Address: 0xdb9e00
//
_QWORD *__fastcall sub_DB9E00(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 *v5; // rsi
  int v6; // r10d
  __int64 v7; // rcx
  unsigned int v8; // eax
  __int64 v9; // r9
  __int64 *v10; // r13
  __int64 v11; // rdx
  char *v12; // r12
  __int64 v13; // r8
  char *v14; // r14
  char *v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // r13
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rsi
  __int64 v23; // rax
  unsigned int v24; // edx
  __int64 v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 *v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rsi
  unsigned int v36; // eax
  __int64 v37; // rbx
  __int64 v38; // rdx
  __int64 v39; // rbx
  __int64 v40; // r12
  __int64 v41; // rdi
  __int64 v42; // rax
  int v43; // edx
  int v44; // r10d
  int v45; // eax
  int v46; // edx
  __int64 v47; // rcx
  __int64 v48; // r8
  _BYTE *v49; // r14
  __int64 v50; // rax
  _BYTE *v51; // r12
  _BYTE *v52; // rdi
  __int64 v53; // rax
  int v54; // r9d
  __int64 v55; // rdi
  __int64 v56; // r10
  unsigned int v57; // eax
  int v58; // r9d
  __int64 v59; // r10
  unsigned int v60; // eax
  _BYTE *v61; // [rsp+8h] [rbp-198h]
  __int64 v62; // [rsp+10h] [rbp-190h]
  char *v63; // [rsp+18h] [rbp-188h]
  char *v64; // [rsp+20h] [rbp-180h] BYREF
  __int64 v65; // [rsp+28h] [rbp-178h]
  _BYTE v66[112]; // [rsp+30h] [rbp-170h] BYREF
  __int128 v67; // [rsp+A0h] [rbp-100h]
  __int128 v68; // [rsp+B0h] [rbp-F0h]
  __int64 v69; // [rsp+C0h] [rbp-E0h] BYREF
  _BYTE *v70; // [rsp+C8h] [rbp-D8h] BYREF
  __int64 v71; // [rsp+D0h] [rbp-D0h] BYREF
  _BYTE v72[104]; // [rsp+D8h] [rbp-C8h] BYREF
  __int64 v73; // [rsp+140h] [rbp-60h]
  __int64 v74; // [rsp+148h] [rbp-58h]
  __int64 v75; // [rsp+150h] [rbp-50h]
  __int64 v76; // [rsp+158h] [rbp-48h]
  char v77; // [rsp+160h] [rbp-40h]

  v4 = a1 + 648;
  v62 = a2;
  v69 = a2;
  v61 = v72;
  v70 = v72;
  v5 = (__int64 *)*(unsigned int *)(a1 + 672);
  v63 = v66;
  v64 = v66;
  v65 = 0x100000000LL;
  v71 = 0x100000000LL;
  v74 = 0;
  LOBYTE(v75) = 0;
  v76 = 0;
  v77 = 0;
  v67 = 0;
  v68 = 0;
  if ( !(_DWORD)v5 )
  {
    ++*(_QWORD *)(a1 + 648);
    goto LABEL_84;
  }
  v6 = 1;
  v7 = *(_QWORD *)(a1 + 656);
  v8 = ((_DWORD)v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = 0;
  v10 = (__int64 *)(v7 + 168LL * v8);
  v11 = *v10;
  if ( a2 == *v10 )
  {
LABEL_3:
    LOBYTE(v61) = 0;
    goto LABEL_4;
  }
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v9 )
      v9 = (__int64)v10;
    v8 = ((_DWORD)v5 - 1) & (v6 + v8);
    v10 = (__int64 *)(v7 + 168LL * v8);
    v11 = *v10;
    if ( v62 == *v10 )
      goto LABEL_3;
    ++v6;
  }
  v45 = *(_DWORD *)(a1 + 664);
  if ( v9 )
    v10 = (__int64 *)v9;
  ++*(_QWORD *)(a1 + 648);
  v46 = v45 + 1;
  if ( 4 * (v45 + 1) >= (unsigned int)(3 * (_DWORD)v5) )
  {
LABEL_84:
    sub_DB6980(v4, 2 * (_DWORD)v5);
    v54 = *(_DWORD *)(a1 + 672);
    if ( v54 )
    {
      v55 = v69;
      v9 = (unsigned int)(v54 - 1);
      v56 = *(_QWORD *)(a1 + 656);
      v5 = (__int64 *)*(unsigned int *)(a1 + 664);
      v57 = v9 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
      v10 = (__int64 *)(v56 + 168LL * v57);
      v46 = (_DWORD)v5 + 1;
      v47 = *v10;
      if ( v69 == *v10 )
        goto LABEL_66;
      v48 = 1;
      v5 = 0;
      while ( v47 != -4096 )
      {
        if ( !v5 && v47 == -8192 )
          v5 = v10;
        v57 = v9 & (v48 + v57);
        v10 = (__int64 *)(v56 + 168LL * v57);
        v47 = *v10;
        if ( v69 == *v10 )
          goto LABEL_66;
        v48 = (unsigned int)(v48 + 1);
      }
LABEL_88:
      v47 = v55;
      if ( v5 )
        v10 = v5;
      goto LABEL_66;
    }
LABEL_113:
    ++*(_DWORD *)(a1 + 664);
    BUG();
  }
  v47 = v62;
  v48 = (unsigned int)v5 >> 3;
  if ( (int)v5 - *(_DWORD *)(a1 + 668) - v46 <= (unsigned int)v48 )
  {
    sub_DB6980(v4, (int)v5);
    v58 = *(_DWORD *)(a1 + 672);
    if ( v58 )
    {
      v9 = (unsigned int)(v58 - 1);
      v55 = v69;
      v59 = *(_QWORD *)(a1 + 656);
      v48 = 1;
      v60 = v9 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
      v10 = (__int64 *)(v59 + 168LL * v60);
      v46 = *(_DWORD *)(a1 + 664) + 1;
      v5 = 0;
      v47 = *v10;
      if ( v69 == *v10 )
        goto LABEL_66;
      while ( v47 != -4096 )
      {
        if ( !v5 && v47 == -8192 )
          v5 = v10;
        v60 = v9 & (v48 + v60);
        v10 = (__int64 *)(v59 + 168LL * v60);
        v47 = *v10;
        if ( v69 == *v10 )
          goto LABEL_66;
        v48 = (unsigned int)(v48 + 1);
      }
      goto LABEL_88;
    }
    goto LABEL_113;
  }
LABEL_66:
  *(_DWORD *)(a1 + 664) = v46;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 668);
  *v10 = v47;
  v10[1] = (__int64)(v10 + 3);
  v10[2] = 0x100000000LL;
  if ( (_DWORD)v71 )
  {
    v5 = (__int64 *)&v70;
    sub_D9EE30((__int64)(v10 + 1), (__int64)&v70, (unsigned int)v71, v47, (unsigned __int64 *)v48, v9);
    v50 = (unsigned int)v71;
    v10[17] = v74;
    *((_BYTE *)v10 + 144) = v75;
    v10[19] = v76;
    *((_BYTE *)v10 + 160) = v77;
    v51 = v70;
    v49 = &v70[112 * v50];
    if ( v70 != v49 )
    {
      do
      {
        v49 -= 112;
        v52 = (_BYTE *)*((_QWORD *)v49 + 8);
        if ( v52 != v49 + 80 )
          _libc_free(v52, &v70);
        if ( v49[32] )
          *((_QWORD *)v49 + 3) = 0;
        v53 = *((_QWORD *)v49 + 3);
        *(_QWORD *)v49 = &unk_49DB368;
        if ( v53 != -4096 && v53 != 0 && v53 != -8192 )
          sub_BD60C0((_QWORD *)v49 + 1);
      }
      while ( v51 != v49 );
      v49 = v70;
    }
  }
  else
  {
    v10[17] = v74;
    *((_BYTE *)v10 + 144) = v75;
    v10[19] = v76;
    *((_BYTE *)v10 + 160) = v77;
    v49 = v70;
  }
  if ( v49 != v61 )
    _libc_free(v49, v5);
  LOBYTE(v61) = 1;
LABEL_4:
  v12 = v64;
  v13 = 112LL * (unsigned int)v65;
  v14 = &v64[v13];
  if ( v64 != &v64[v13] )
  {
    do
    {
      v14 -= 112;
      v15 = (char *)*((_QWORD *)v14 + 8);
      if ( v15 != v14 + 80 )
        _libc_free(v15, v5);
      if ( v14[32] )
        *((_QWORD *)v14 + 3) = 0;
      v16 = *((_QWORD *)v14 + 3);
      *(_QWORD *)v14 = &unk_49DB368;
      if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
        sub_BD60C0((_QWORD *)v14 + 1);
    }
    while ( v12 != v14 );
    v14 = v64;
  }
  if ( v14 != v63 )
    _libc_free(v14, v5);
  v17 = v10 + 1;
  if ( (_BYTE)v61 )
  {
    sub_DB9040((__int64)&v69, a1, v62, 0);
    if ( !(_DWORD)v70 && sub_D96A50(v73) )
      goto LABEL_34;
    v22 = *(_QWORD *)(a1 + 1168);
    v64 = v63;
    v65 = 0x800000000LL;
    v23 = *(unsigned int *)(a1 + 1184);
    if ( (_DWORD)v23 )
    {
      v20 = (unsigned int)(v23 - 1);
      v24 = v20 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
      v19 = v22 + 56LL * v24;
      v25 = *(_QWORD *)v19;
      if ( v62 == *(_QWORD *)v19 )
      {
LABEL_21:
        if ( v19 != v22 + 56 * v23 )
        {
          sub_D927B0((__int64)&v64, v63, *(_QWORD *)(v19 + 8), *(_QWORD *)(v19 + 8) + 8LL * *(unsigned int *)(v19 + 16));
          v26 = (__int64)v64;
          v27 = (unsigned int)v65;
LABEL_23:
          sub_DAB940(a1, v26, v27, v19, v20, v21);
          v28 = sub_AA5930(**(_QWORD **)(v62 + 32));
          v20 = v29;
          while ( v20 != v28 )
          {
            v26 = *(unsigned int *)(a1 + 768);
            v30 = *(_QWORD *)(a1 + 752);
            if ( (_DWORD)v26 )
            {
              v26 = (unsigned int)(v26 - 1);
              v31 = v26 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
              v32 = (__int64 *)(v30 + 16LL * v31);
              v21 = *v32;
              if ( *v32 == v28 )
              {
LABEL_26:
                *v32 = -8192;
                --*(_DWORD *)(a1 + 760);
                ++*(_DWORD *)(a1 + 764);
              }
              else
              {
                v43 = 1;
                while ( v21 != -4096 )
                {
                  v44 = v43 + 1;
                  v31 = v26 & (v43 + v31);
                  v32 = (__int64 *)(v30 + 16LL * v31);
                  v21 = *v32;
                  if ( v28 == *v32 )
                    goto LABEL_26;
                  v43 = v44;
                }
              }
            }
            if ( !v28 )
              BUG();
            v33 = *(_QWORD *)(v28 + 32);
            if ( !v33 )
              BUG();
            v28 = 0;
            if ( *(_BYTE *)(v33 - 24) == 84 )
              v28 = v33 - 24;
          }
          if ( v64 != v63 )
            _libc_free(v64, v26);
LABEL_34:
          v34 = *(unsigned int *)(a1 + 672);
          v35 = *(_QWORD *)(a1 + 656);
          if ( (_DWORD)v34 )
          {
            v36 = (v34 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
            v20 = 5LL * v36;
            v37 = v35 + 168LL * v36;
            v38 = *(_QWORD *)v37;
            if ( v62 == *(_QWORD *)v37 )
            {
LABEL_36:
              v17 = (_QWORD *)(v37 + 8);
              sub_D9EE30(v37 + 8, (__int64)&v69, v38, v34, (unsigned __int64 *)v20, v21);
              *(_QWORD *)(v37 + 136) = v73;
              *(_BYTE *)(v37 + 144) = v74;
              *(_QWORD *)(v37 + 152) = v75;
              *(_BYTE *)(v37 + 160) = v76;
              v39 = v69;
              v40 = v69 + 112LL * (unsigned int)v70;
              if ( v69 != v40 )
              {
                do
                {
                  v40 -= 112;
                  v41 = *(_QWORD *)(v40 + 64);
                  if ( v41 != v40 + 80 )
                    _libc_free(v41, &v69);
                  if ( *(_BYTE *)(v40 + 32) )
                    *(_QWORD *)(v40 + 24) = 0;
                  v42 = *(_QWORD *)(v40 + 24);
                  *(_QWORD *)v40 = &unk_49DB368;
                  if ( v42 != -4096 && v42 != 0 && v42 != -8192 )
                    sub_BD60C0((_QWORD *)(v40 + 8));
                }
                while ( v39 != v40 );
                v40 = v69;
              }
              if ( (__int64 *)v40 != &v71 )
                _libc_free(v40, &v69);
              return v17;
            }
            v20 = 1;
            while ( v38 != -4096 )
            {
              v21 = (unsigned int)(v20 + 1);
              v36 = (v34 - 1) & (v20 + v36);
              v20 = 5LL * v36;
              v37 = v35 + 168LL * v36;
              v38 = *(_QWORD *)v37;
              if ( v62 == *(_QWORD *)v37 )
                goto LABEL_36;
              v20 = (unsigned int)v21;
            }
          }
          v38 = 5LL * (unsigned int)v34;
          v37 = v35 + 168LL * (unsigned int)v34;
          goto LABEL_36;
        }
      }
      else
      {
        v19 = 1;
        while ( v25 != -4096 )
        {
          v21 = (unsigned int)(v19 + 1);
          v24 = v20 & (v19 + v24);
          v19 = v22 + 56LL * v24;
          v25 = *(_QWORD *)v19;
          if ( v62 == *(_QWORD *)v19 )
            goto LABEL_21;
          v19 = (unsigned int)v21;
        }
      }
    }
    v26 = (__int64)v63;
    v27 = 0;
    goto LABEL_23;
  }
  return v17;
}
