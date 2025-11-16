// Function: sub_1B570D0
// Address: 0x1b570d0
//
__int64 __fastcall sub_1B570D0(__int64 a1)
{
  __int64 *v1; // r13
  int v2; // eax
  __int64 v3; // rbx
  unsigned int v4; // ecx
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rdi
  __int64 v13; // rax
  char v14; // r8
  unsigned int v15; // esi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rcx
  _QWORD *v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // r14
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r11
  __int64 v28; // r10
  char v29; // di
  unsigned int v30; // esi
  __int64 v31; // rdx
  __int64 v32; // rax
  int v33; // r14d
  __int64 v34; // rcx
  __int64 v35; // rax
  int v36; // r9d
  __int64 v37; // rcx
  __int64 *v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rdx
  int *v41; // rdx
  __int64 v42; // rsi
  __int64 *v43; // rbx
  unsigned __int64 v44; // rdi
  _QWORD *v46; // rax
  _QWORD *v47; // r9
  __int64 v48; // rcx
  __int64 v49; // rsi
  __int64 *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rsi
  unsigned __int64 v53; // rcx
  __int64 v54; // rsi
  __int64 v55; // rax
  int v56; // r12d
  __int64 *v57; // r11
  int v58; // ecx
  __int64 *v59; // r8
  __int64 *v60; // rdx
  __int64 v61; // rcx
  __int64 *v62; // rax
  unsigned __int64 v63; // rdx
  unsigned int *v64; // rdi
  unsigned int *v65; // r10
  __int64 v66; // rsi
  __int64 v67; // r9
  _QWORD *v68; // rdx
  __int64 v69; // r11
  unsigned __int64 v70; // r9
  __int64 v71; // r9
  __int64 v72; // rdx
  __int64 v73; // r8
  int v74; // edi
  __int64 *v75; // rsi
  __int64 v76; // rdx
  int v77; // edi
  __int64 v78; // r8
  __int64 *v79; // [rsp+8h] [rbp-78h]
  __int64 v80; // [rsp+8h] [rbp-78h]
  __int64 v81; // [rsp+8h] [rbp-78h]
  __int64 v82; // [rsp+10h] [rbp-70h]
  unsigned int v83; // [rsp+18h] [rbp-68h]
  unsigned __int8 v84; // [rsp+1Fh] [rbp-61h]
  __int64 v86; // [rsp+28h] [rbp-58h]
  __int64 v87; // [rsp+30h] [rbp-50h] BYREF
  __int64 *v88; // [rsp+38h] [rbp-48h]
  __int64 v89; // [rsp+40h] [rbp-40h]
  unsigned int v90; // [rsp+48h] [rbp-38h]

  v1 = 0;
  v2 = *(_DWORD *)(a1 + 20);
  v84 = 0;
  v87 = 0;
  v3 = *(_QWORD *)(a1 + 40);
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v82 = ((v2 & 0xFFFFFFFu) >> 1) - 1;
  v86 = 0;
  if ( (v2 & 0xFFFFFFFu) >> 1 == 1 )
    goto LABEL_48;
  do
  {
    v4 = 2 * ++v86;
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v5 = *(_QWORD *)(a1 - 8);
    else
      v5 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v6 = *(_QWORD *)(v5 + 24LL * v4);
    v7 = 24;
    if ( (_DWORD)v86 != -1 )
      v7 = 24LL * (v4 + 1);
    v8 = *(_QWORD *)(v5 + v7);
    v9 = sub_157F280(v8);
    v11 = v10;
    v12 = v9;
    while ( v11 != v12 )
    {
      v13 = 0x17FFFFFFE8LL;
      v14 = *(_BYTE *)(v12 + 23) & 0x40;
      v15 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
      if ( v15 )
      {
        v16 = 24LL * *(unsigned int *)(v12 + 56) + 8;
        v17 = 0;
        do
        {
          v18 = v12 - 24LL * v15;
          if ( v14 )
            v18 = *(_QWORD *)(v12 - 8);
          if ( v3 == *(_QWORD *)(v18 + v16) )
          {
            v13 = 24 * v17;
            goto LABEL_14;
          }
          ++v17;
          v16 += 8;
        }
        while ( v15 != (_DWORD)v17 );
        v13 = 0x17FFFFFFE8LL;
      }
LABEL_14:
      if ( v14 )
      {
        v19 = *(_QWORD *)(v12 - 8);
        v20 = (_QWORD *)(v19 + v13);
        v21 = *(_QWORD *)(v19 + v13);
        if ( v6 == v21 )
          goto LABEL_50;
      }
      else
      {
        v19 = v12 - 24LL * v15;
        v20 = (_QWORD *)(v19 + v13);
        v21 = *(_QWORD *)(v19 + v13);
        if ( v6 == v21 )
        {
LABEL_50:
          v46 = (_QWORD *)(v19 + 24LL * *(unsigned int *)(v12 + 56) + 8);
          v47 = &v46[v15];
          if ( v47 != v46 )
          {
            v48 = 0;
            do
            {
              v49 = v3 == *v46++;
              v48 += v49;
            }
            while ( v46 != v47 );
            if ( v48 == 1 )
            {
              if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
                v50 = *(__int64 **)(a1 - 8);
              else
                v50 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
              v51 = *v50;
              if ( v21 )
              {
                v52 = v20[1];
                v53 = v20[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v53 = v52;
                if ( v52 )
                  *(_QWORD *)(v52 + 16) = *(_QWORD *)(v52 + 16) & 3LL | v53;
              }
              *v20 = v51;
              v84 = 1;
              if ( v51 )
              {
                v54 = *(_QWORD *)(v51 + 8);
                v20[1] = v54;
                if ( v54 )
                  *(_QWORD *)(v54 + 16) = (unsigned __int64)(v20 + 1) | *(_QWORD *)(v54 + 16) & 3LL;
                v84 = 1;
                v20[2] = v20[2] & 3LL | (v51 + 8);
                *(_QWORD *)(v51 + 8) = v20;
              }
            }
          }
        }
      }
      v22 = *(_QWORD *)(v12 + 32);
      if ( !v22 )
        BUG();
      v12 = 0;
      if ( *(_BYTE *)(v22 - 8) == 77 )
        v12 = v22 - 24;
    }
    v23 = sub_157ED60(v8);
    if ( v23 != sub_157EBA0(v8) )
      continue;
    if ( !sub_157F0B0(v8) )
      continue;
    v24 = sub_157EBA0(v8);
    if ( *(_BYTE *)(v24 + 16) != 26 )
      continue;
    if ( (*(_DWORD *)(v24 + 20) & 0xFFFFFFF) != 1 )
      continue;
    v25 = sub_157F280(*(_QWORD *)(v24 - 24));
    v27 = v26;
    v28 = v25;
    if ( v25 == v26 )
      continue;
    while ( 1 )
    {
      v29 = *(_BYTE *)(v28 + 23) & 0x40;
      v30 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
      if ( v30 )
      {
        v31 = 24LL * *(unsigned int *)(v28 + 56) + 8;
        v32 = 0;
        while ( 1 )
        {
          v33 = v32;
          v34 = v28 - 24LL * v30;
          if ( v29 )
            v34 = *(_QWORD *)(v28 - 8);
          if ( v8 == *(_QWORD *)(v34 + v31) )
            break;
          ++v32;
          v31 += 8;
          if ( v30 == (_DWORD)v32 )
            goto LABEL_66;
        }
        v35 = 24 * v32;
        if ( v29 )
        {
LABEL_32:
          if ( v6 == *(_QWORD *)(*(_QWORD *)(v28 - 8) + v35) )
            break;
          goto LABEL_68;
        }
      }
      else
      {
LABEL_66:
        v35 = 0x17FFFFFFE8LL;
        v33 = -1;
        if ( v29 )
          goto LABEL_32;
      }
      if ( v6 == *(_QWORD *)(v28 - 24LL * v30 + v35) )
        break;
LABEL_68:
      v55 = *(_QWORD *)(v28 + 32);
      if ( !v55 )
        BUG();
      v28 = 0;
      if ( *(_BYTE *)(v55 - 8) == 77 )
        v28 = v55 - 24;
      if ( v27 == v28 )
        goto LABEL_38;
    }
    if ( !v90 )
    {
      ++v87;
LABEL_113:
      v80 = v28;
      sub_1B56DE0((__int64)&v87, 2 * v90);
      if ( v90 )
      {
        v28 = v80;
        v58 = v89 + 1;
        LODWORD(v72) = (v90 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
        v38 = &v88[5 * (unsigned int)v72];
        v73 = *v38;
        if ( *v38 == v80 )
          goto LABEL_80;
        v74 = 1;
        v75 = 0;
        while ( v73 != -8 )
        {
          if ( !v75 && v73 == -16 )
            v75 = v38;
          v72 = (v90 - 1) & ((_DWORD)v72 + v74);
          v38 = &v88[5 * v72];
          v73 = *v38;
          if ( *v38 == v80 )
            goto LABEL_80;
          ++v74;
        }
LABEL_125:
        if ( v75 )
          v38 = v75;
        goto LABEL_80;
      }
LABEL_140:
      LODWORD(v89) = v89 + 1;
      BUG();
    }
    v36 = v90 - 1;
    LODWORD(v37) = (v90 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v38 = &v88[5 * (unsigned int)v37];
    v39 = *v38;
    if ( *v38 == v28 )
    {
LABEL_35:
      v40 = *((unsigned int *)v38 + 4);
      if ( (unsigned int)v40 >= *((_DWORD *)v38 + 5) )
      {
        v79 = v38;
        sub_16CD150((__int64)(v38 + 1), v38 + 3, 0, 4, (int)v88, v36);
        v38 = v79;
        v41 = (int *)(v79[1] + 4LL * *((unsigned int *)v79 + 4));
      }
      else
      {
        v41 = (int *)(v38[1] + 4 * v40);
      }
      goto LABEL_37;
    }
    v56 = 1;
    v57 = 0;
    while ( v39 != -8 )
    {
      if ( v39 == -16 && !v57 )
        v57 = v38;
      v37 = v36 & (unsigned int)(v37 + v56);
      v38 = &v88[5 * v37];
      v39 = *v38;
      if ( *v38 == v28 )
        goto LABEL_35;
      ++v56;
    }
    if ( v57 )
      v38 = v57;
    ++v87;
    v58 = v89 + 1;
    if ( 4 * ((int)v89 + 1) >= 3 * v90 )
      goto LABEL_113;
    if ( v90 - HIDWORD(v89) - v58 <= v90 >> 3 )
    {
      v83 = ((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4);
      v81 = v28;
      sub_1B56DE0((__int64)&v87, v90);
      if ( v90 )
      {
        v75 = 0;
        v28 = v81;
        LODWORD(v76) = (v90 - 1) & v83;
        v58 = v89 + 1;
        v77 = 1;
        v38 = &v88[5 * (unsigned int)v76];
        v78 = *v38;
        if ( v81 == *v38 )
          goto LABEL_80;
        while ( v78 != -8 )
        {
          if ( v78 == -16 && !v75 )
            v75 = v38;
          v76 = (v90 - 1) & ((_DWORD)v76 + v77);
          v38 = &v88[5 * v76];
          v78 = *v38;
          if ( *v38 == v81 )
            goto LABEL_80;
          ++v77;
        }
        goto LABEL_125;
      }
      goto LABEL_140;
    }
LABEL_80:
    LODWORD(v89) = v58;
    if ( *v38 != -8 )
      --HIDWORD(v89);
    v41 = (int *)(v38 + 3);
    *v38 = v28;
    v38[1] = (__int64)(v38 + 3);
    v38[2] = 0x400000000LL;
LABEL_37:
    *v41 = v33;
    ++*((_DWORD *)v38 + 4);
LABEL_38:
    ;
  }
  while ( v82 != v86 );
  v1 = v88;
  v42 = v90;
  if ( !(_DWORD)v89 )
    goto LABEL_40;
  v59 = &v88[5 * v90];
  if ( v88 == v59 )
    goto LABEL_40;
  v60 = v88;
  while ( 1 )
  {
    v61 = *v60;
    v62 = v60;
    if ( *v60 != -8 && v61 != -16 )
      break;
    v60 += 5;
    if ( v59 == v60 )
      goto LABEL_40;
  }
  if ( v59 != v60 )
  {
    while ( 1 )
    {
      v63 = *((unsigned int *)v62 + 4);
      if ( v63 > 1 )
        break;
LABEL_105:
      v62 += 5;
      if ( v62 == v59 )
        goto LABEL_109;
      while ( *v62 == -16 || *v62 == -8 )
      {
        v62 += 5;
        if ( v59 == v62 )
          goto LABEL_109;
      }
      if ( v59 == v62 )
      {
LABEL_109:
        v42 = v90;
        v1 = v88;
        goto LABEL_40;
      }
      v61 = *v62;
    }
    v64 = (unsigned int *)v62[1];
    v65 = &v64[v63];
    while ( 2 )
    {
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      {
        v66 = **(_QWORD **)(a1 - 8);
        if ( (*(_BYTE *)(v61 + 23) & 0x40) == 0 )
          goto LABEL_103;
      }
      else
      {
        v66 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
        if ( (*(_BYTE *)(v61 + 23) & 0x40) == 0 )
        {
LABEL_103:
          v67 = v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF);
          goto LABEL_93;
        }
      }
      v67 = *(_QWORD *)(v61 - 8);
LABEL_93:
      v68 = (_QWORD *)(v67 + 24LL * *v64);
      if ( *v68 )
      {
        v69 = v68[1];
        v70 = v68[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v70 = v69;
        if ( v69 )
          *(_QWORD *)(v69 + 16) = *(_QWORD *)(v69 + 16) & 3LL | v70;
      }
      *v68 = v66;
      if ( v66 )
      {
        v71 = *(_QWORD *)(v66 + 8);
        v68[1] = v71;
        if ( v71 )
          *(_QWORD *)(v71 + 16) = (unsigned __int64)(v68 + 1) | *(_QWORD *)(v71 + 16) & 3LL;
        v68[2] = (v66 + 8) | v68[2] & 3LL;
        *(_QWORD *)(v66 + 8) = v68;
      }
      if ( v65 == ++v64 )
      {
        v84 = 1;
        goto LABEL_105;
      }
      continue;
    }
  }
LABEL_40:
  if ( (_DWORD)v42 )
  {
    v43 = &v1[5 * v42];
    do
    {
      if ( *v1 != -16 && *v1 != -8 )
      {
        v44 = v1[1];
        if ( (__int64 *)v44 != v1 + 3 )
          _libc_free(v44);
      }
      v1 += 5;
    }
    while ( v43 != v1 );
    v1 = v88;
  }
LABEL_48:
  j___libc_free_0(v1);
  return v84;
}
