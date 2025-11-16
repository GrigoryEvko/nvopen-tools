// Function: sub_288D800
// Address: 0x288d800
//
__int64 __fastcall sub_288D800(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  int v11; // r11d
  unsigned int i; // eax
  __int64 v13; // rcx
  unsigned int v14; // eax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 j; // r9
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  int v24; // r15d
  __int64 v25; // rdx
  int v26; // r9d
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 v29; // r14
  __int64 v30; // r12
  char v31; // bl
  __int64 v32; // r8
  __int64 v33; // r9
  char v34; // r12
  unsigned __int64 *v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  int v41; // eax
  unsigned __int64 v42; // rsi
  __int64 v43; // r12
  __int64 *v44; // r8
  int v45; // edi
  unsigned int v46; // ecx
  __int64 *v47; // rdx
  __int64 v48; // r9
  __int64 v49; // rax
  __int64 *v50; // rdx
  __int64 v51; // rcx
  unsigned __int16 v52; // bx
  __int64 v53; // rdx
  char *v54; // rsi
  __int64 v55; // rdi
  unsigned int v56; // ebx
  int v57; // edx
  int v58; // r10d
  __int64 **v59; // [rsp+10h] [rbp-180h]
  __int64 v60; // [rsp+20h] [rbp-170h]
  __int64 v62; // [rsp+30h] [rbp-160h]
  __int64 *v63; // [rsp+58h] [rbp-138h]
  __int64 v64; // [rsp+60h] [rbp-130h]
  __int64 *v65; // [rsp+68h] [rbp-128h]
  __int64 v66; // [rsp+70h] [rbp-120h]
  char v67; // [rsp+70h] [rbp-120h]
  __int64 *v68; // [rsp+78h] [rbp-118h]
  __int64 v69; // [rsp+80h] [rbp-110h]
  __int64 *v70; // [rsp+88h] [rbp-108h]
  char v71; // [rsp+90h] [rbp-100h]
  __int64 v72; // [rsp+98h] [rbp-F8h]
  __int64 v73; // [rsp+A0h] [rbp-F0h]
  __int64 v74; // [rsp+A8h] [rbp-E8h]
  __int64 v75[2]; // [rsp+B0h] [rbp-E0h] BYREF
  _QWORD v76[2]; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v77; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v78; // [rsp+D8h] [rbp-B8h]
  __int64 *v79; // [rsp+E0h] [rbp-B0h] BYREF
  unsigned int v80; // [rsp+E8h] [rbp-A8h]
  _BYTE *v81; // [rsp+120h] [rbp-70h] BYREF
  __int64 v82; // [rsp+128h] [rbp-68h]
  _BYTE v83[48]; // [rsp+130h] [rbp-60h] BYREF
  char v84; // [rsp+160h] [rbp-30h] BYREF

  v7 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  if ( *(_QWORD *)(v7 + 40) == *(_QWORD *)(v7 + 48) )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  else
  {
    v68 = (__int64 *)(sub_BC1CD0(a4, &unk_4F881D0, a3) + 8);
    v59 = (__int64 **)(sub_BC1CD0(a4, &unk_4F89C30, a3) + 8);
    v69 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
    v72 = sub_BC1CD0(a4, &unk_4F86630, a3) + 8;
    v63 = (__int64 *)(sub_BC1CD0(a4, &unk_4F8FAE8, a3) + 8);
    v64 = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
    v8 = sub_BC1CD0(a4, &unk_4F8FC88, a3);
    v9 = *(unsigned int *)(a4 + 88);
    v10 = *(_QWORD *)(a4 + 72);
    v65 = (__int64 *)(v8 + 8);
    if ( !(_DWORD)v9 )
      goto LABEL_71;
    v11 = 1;
    for ( i = (v9 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4FDBCE0 >> 9) ^ ((unsigned int)&unk_4FDBCE0 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v9 - 1) & v14 )
    {
      v13 = v10 + 24LL * i;
      if ( *(_UNKNOWN **)v13 == &unk_4FDBCE0 && a3 == *(_QWORD *)(v13 + 8) )
        break;
      if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == -4096 )
        goto LABEL_71;
      v14 = v11 + i;
      ++v11;
    }
    if ( v13 == v10 + 24 * v9 )
    {
LABEL_71:
      v62 = 0;
    }
    else
    {
      v16 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL);
      v62 = v16;
      if ( v16 )
        v62 = *(_QWORD *)(v16 + 8);
    }
    v17 = sub_BC1CD0(a4, &unk_4F82410, a3);
    v20 = *(_QWORD *)(a3 + 40);
    v21 = *(_QWORD *)(v17 + 8);
    v22 = *(_QWORD *)(v21 + 72);
    v23 = *(unsigned int *)(v21 + 88);
    if ( !(_DWORD)v23 )
      goto LABEL_69;
    v24 = 1;
    v18 = (unsigned int)(v23 - 1);
    for ( j = (unsigned int)v18
            & ((unsigned int)((0xBF58476D1CE4E5B9LL
                             * (((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32)
                              | ((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4))) >> 31)
             ^ (484763065 * (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4)))); ; j = (unsigned int)v18 & v26 )
    {
      v25 = v22 + 24LL * (unsigned int)j;
      if ( *(_UNKNOWN **)v25 == &unk_4F87C68 && v20 == *(_QWORD *)(v25 + 8) )
        break;
      if ( *(_QWORD *)v25 == -4096 && *(_QWORD *)(v25 + 8) == -4096 )
        goto LABEL_69;
      v26 = v24 + j;
      ++v24;
    }
    if ( v25 == v22 + 24 * v23 )
    {
LABEL_69:
      v60 = 0;
      v70 = 0;
    }
    else
    {
      v27 = *(__int64 **)(*(_QWORD *)(v25 + 16) + 24LL);
      v70 = v27;
      if ( v27 )
      {
        v78 = 1;
        v60 = (__int64)(v27 + 1);
        v28 = (__int64 *)&v79;
        do
        {
          *v28 = -4096;
          v28 += 2;
        }
        while ( v28 != (__int64 *)&v84 );
        if ( (v78 & 1) == 0 )
          sub_C7D6A0((__int64)v79, 16LL * v80, 8);
        v70 = (__int64 *)v70[2];
        if ( v70 )
          v70 = (__int64 *)(sub_BC1CD0(a4, &unk_4F8D9A8, a3) + 8);
      }
      else
      {
        v60 = 0;
      }
    }
    v29 = v7 + 8;
    v66 = *(_QWORD *)(v7 + 48);
    if ( v66 == *(_QWORD *)(v7 + 40) )
    {
      v34 = 0;
    }
    else
    {
      v71 = 0;
      v30 = *(_QWORD *)(v7 + 40);
      do
      {
        v30 += 8;
        v31 = sub_F6AC10(*(char **)(v30 - 8), v69, v29, (__int64)v68, v72, 0, 0);
        v71 |= sub_11D2180(*(_QWORD *)(v30 - 8), v69, v29, (__int64)v68, v32, v33) | v31;
      }
      while ( v66 != v30 );
      v34 = v71;
    }
    v77 = 0;
    v35 = (unsigned __int64 *)&v79;
    v78 = 1;
    do
    {
      *v35 = -4096;
      v35 += 2;
    }
    while ( v35 != (unsigned __int64 *)&v81 );
    v36 = (__int64)&v77;
    v81 = v83;
    v82 = 0x400000000LL;
    sub_F774D0(v29, (__int64)&v77, (__int64)&v81, v20, v18, j);
    v41 = v82;
    if ( (_DWORD)v82 )
    {
      v67 = v34;
      while ( 1 )
      {
        v42 = (unsigned __int64)v81;
        v43 = *(_QWORD *)&v81[8 * v41 - 8];
        if ( (v78 & 1) != 0 )
          break;
        v44 = v79;
        if ( v80 )
        {
          v45 = v80 - 1;
LABEL_37:
          v46 = v45 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          v47 = &v44[2 * v46];
          v48 = *v47;
          if ( *v47 == v43 )
          {
LABEL_38:
            *v47 = -8192;
            ++HIDWORD(v78);
            v42 = (unsigned __int64)v81;
            LODWORD(v78) = (2 * ((unsigned int)v78 >> 1) - 2) | v78 & 1;
            v41 = v82;
          }
          else
          {
            v57 = 1;
            while ( v48 != -4096 )
            {
              v58 = v57 + 1;
              v46 = v45 & (v57 + v46);
              v47 = &v44[2 * v46];
              v48 = *v47;
              if ( v43 == *v47 )
                goto LABEL_38;
              v57 = v58;
            }
          }
        }
        v49 = (unsigned int)(v41 - 1);
        v50 = (__int64 *)(v42 + 8 * v49 - 8);
        do
        {
          LODWORD(v82) = v49;
          if ( !(_DWORD)v49 )
            break;
          v51 = *v50;
          LODWORD(v49) = v49 - 1;
          --v50;
        }
        while ( !v51 );
        v52 = *(_WORD *)(a2 + 2);
        if ( v60 && (unsigned __int8)sub_D84420(v60) )
          v52 = 256;
        v53 = 14;
        v54 = "<unnamed loop>";
        v55 = **(_QWORD **)(v43 + 32);
        if ( v55 && (*(_BYTE *)(v55 + 7) & 0x10) != 0 )
          v54 = (char *)sub_BD5D20(v55);
        v75[0] = (__int64)v76;
        sub_287ECD0(v75, v54, (__int64)&v54[v53]);
        BYTE4(v74) = 0;
        BYTE4(v73) = 0;
        v56 = sub_288A700(
                v43,
                v69,
                v29,
                v68,
                v59,
                v72,
                v63,
                v70,
                v60,
                v65,
                1u,
                *(_DWORD *)(a2 + 20),
                0,
                *(_BYTE *)(a2 + 24),
                *(_BYTE *)(a2 + 25),
                v73,
                v74,
                *(_WORD *)a2,
                *(_WORD *)(a2 + 4),
                *(_WORD *)(a2 + 6),
                v52,
                *(_WORD *)(a2 + 8),
                *(_QWORD *)(a2 + 12),
                v64);
        v36 = v56;
        sub_DFA060((__int64)v59);
        v67 |= v56 != 0;
        if ( v56 == 2 && v62 )
        {
          v36 = v43;
          sub_22D0060(v62, v43, v75[0], v75[1]);
        }
        if ( (_QWORD *)v75[0] != v76 )
        {
          v36 = v76[0] + 1LL;
          j_j___libc_free_0(v75[0]);
        }
        v41 = v82;
        if ( !(_DWORD)v82 )
        {
          v34 = v67;
          goto LABEL_55;
        }
      }
      v44 = (__int64 *)&v79;
      v45 = 3;
      goto LABEL_37;
    }
LABEL_55:
    if ( v34 )
    {
      sub_22D0390(a1, v36, v37, v38, v39, v40);
    }
    else
    {
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 8) = a1 + 32;
      *(_QWORD *)(a1 + 56) = a1 + 80;
      *(_QWORD *)(a1 + 16) = 0x100000002LL;
      *(_QWORD *)(a1 + 64) = 2;
      *(_DWORD *)(a1 + 72) = 0;
      *(_BYTE *)(a1 + 76) = 1;
      *(_DWORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)(a1 + 32) = &qword_4F82400;
      *(_QWORD *)a1 = 1;
    }
    if ( v81 != v83 )
      _libc_free((unsigned __int64)v81);
    if ( (v78 & 1) == 0 )
      sub_C7D6A0((__int64)v79, 16LL * v80, 8);
  }
  return a1;
}
