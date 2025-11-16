// Function: sub_25DF290
// Address: 0x25df290
//
void __fastcall sub_25DF290(__int64 a1)
{
  __int64 v1; // rbx
  __int64 *v2; // rax
  __int64 v3; // rax
  signed __int64 v4; // r12
  unsigned __int8 **v5; // rax
  unsigned __int8 *v6; // r12
  int v7; // eax
  unsigned __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rcx
  int v12; // edx
  unsigned __int64 *v13; // rax
  __int64 v14; // rbx
  _QWORD *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r13
  _QWORD *i; // rbx
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // rax
  size_t v25; // rdx
  __int64 v26; // rbx
  unsigned __int64 *v27; // r12
  unsigned __int64 v28; // rdi
  __int64 v29; // rdx
  unsigned __int64 *v30; // rbx
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  unsigned __int64 v33; // rbx
  unsigned __int8 *v34; // r13
  __int64 v35; // rax
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // r8
  __int64 v39; // r9
  _QWORD *v40; // rax
  _BYTE *v41; // rdx
  _QWORD *j; // rdx
  __int64 v43; // rbx
  __int64 v44; // rax
  signed __int64 v45; // r13
  _BYTE *v46; // rax
  unsigned __int64 v47; // rbx
  _QWORD *v48; // r12
  __int64 v49; // rdx
  _QWORD *v50; // rax
  __int64 v51; // r14
  __int64 v52; // rax
  unsigned int v53; // r13d
  __int64 v54; // rax
  __int64 *v55; // r15
  __int64 v56; // rax
  __int64 v57; // rax
  _QWORD *v58; // rax
  unsigned __int64 v59; // r15
  _BYTE *v60; // r13
  __int64 v61; // rdx
  unsigned int v62; // esi
  unsigned __int8 **v63; // [rsp+38h] [rbp-228h]
  unsigned __int8 v64; // [rsp+57h] [rbp-209h]
  __int64 v65; // [rsp+58h] [rbp-208h]
  unsigned __int8 **v66; // [rsp+60h] [rbp-200h]
  _BYTE *v67; // [rsp+68h] [rbp-1F8h]
  __int64 v68; // [rsp+78h] [rbp-1E8h]
  _BYTE *v69; // [rsp+80h] [rbp-1E0h] BYREF
  __int64 v70; // [rsp+88h] [rbp-1D8h]
  _BYTE v71[16]; // [rsp+90h] [rbp-1D0h] BYREF
  _BYTE *v72; // [rsp+A0h] [rbp-1C0h] BYREF
  __int64 v73; // [rsp+A8h] [rbp-1B8h]
  _BYTE v74[16]; // [rsp+B0h] [rbp-1B0h] BYREF
  const char *v75; // [rsp+C0h] [rbp-1A0h] BYREF
  char v76; // [rsp+E0h] [rbp-180h]
  char v77; // [rsp+E1h] [rbp-17Fh]
  __int64 v78[4]; // [rsp+F0h] [rbp-170h] BYREF
  __int16 v79; // [rsp+110h] [rbp-150h]
  unsigned __int8 **v80; // [rsp+120h] [rbp-140h] BYREF
  __int64 v81; // [rsp+128h] [rbp-138h]
  _BYTE v82[32]; // [rsp+130h] [rbp-130h] BYREF
  unsigned __int64 *v83; // [rsp+150h] [rbp-110h] BYREF
  __int64 v84; // [rsp+158h] [rbp-108h]
  _BYTE v85[64]; // [rsp+160h] [rbp-100h] BYREF
  _BYTE *v86; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 v87; // [rsp+1A8h] [rbp-B8h]
  _BYTE v88[32]; // [rsp+1B0h] [rbp-B0h] BYREF
  __int64 v89; // [rsp+1D0h] [rbp-90h]
  __int64 v90; // [rsp+1D8h] [rbp-88h]
  __int64 v91; // [rsp+1E0h] [rbp-80h]
  __int64 *v92; // [rsp+1E8h] [rbp-78h]
  void **v93; // [rsp+1F0h] [rbp-70h]
  void **v94; // [rsp+1F8h] [rbp-68h]
  __int64 v95; // [rsp+200h] [rbp-60h]
  int v96; // [rsp+208h] [rbp-58h]
  __int16 v97; // [rsp+20Ch] [rbp-54h]
  char v98; // [rsp+20Eh] [rbp-52h]
  __int64 v99; // [rsp+210h] [rbp-50h]
  __int64 v100; // [rsp+218h] [rbp-48h]
  void *v101; // [rsp+220h] [rbp-40h] BYREF
  void *v102; // [rsp+228h] [rbp-38h] BYREF

  sub_25DCA60(a1, 84);
  v1 = *(_QWORD *)(a1 + 16);
  v2 = **(__int64 ***)(a1 + 40);
  v86 = v88;
  v92 = v2;
  v93 = &v101;
  v94 = &v102;
  v87 = 0x200000000LL;
  v95 = 0;
  v101 = &unk_49DA100;
  v96 = 0;
  v97 = 512;
  v102 = &unk_49DA0B0;
  v80 = (unsigned __int8 **)v82;
  v98 = 7;
  v99 = 0;
  v100 = 0;
  v89 = 0;
  v90 = 0;
  LOWORD(v91) = 0;
  v81 = 0x400000000LL;
  if ( v1 )
  {
    v3 = v1;
    v4 = 0;
    do
    {
      v3 = *(_QWORD *)(v3 + 8);
      ++v4;
    }
    while ( v3 );
    v5 = (unsigned __int8 **)v82;
    if ( v4 > 4 )
    {
      sub_C8D5F0((__int64)&v80, v82, v4, 8u, 512, 0);
      v5 = &v80[(unsigned int)v81];
    }
    do
    {
      *v5++ = *(unsigned __int8 **)(v1 + 24);
      v1 = *(_QWORD *)(v1 + 8);
    }
    while ( v1 );
    LODWORD(v81) = v81 + v4;
    v63 = &v80[(unsigned int)v81];
    if ( v80 != v63 )
    {
      v66 = v80;
      while ( 1 )
      {
        v6 = *v66;
        v7 = **v66;
        if ( (unsigned __int8)v7 <= 0x1Cu )
          goto LABEL_47;
        v8 = (unsigned int)(v7 - 34);
        if ( (unsigned __int8)v8 > 0x33u )
          goto LABEL_47;
        v9 = 0x8000000000041LL;
        if ( !_bittest64(&v9, v8) )
          goto LABEL_47;
        v10 = (__int64)*v66;
        v83 = (unsigned __int64 *)v85;
        v84 = 0x100000000LL;
        sub_B56970(v10, (__int64)&v83);
        v11 = (__int64)v83;
        v12 = v84;
        v13 = &v83[7 * (unsigned int)v84];
        if ( v83 == v13 )
LABEL_94:
          BUG();
        v14 = (__int64)v83;
        while ( 1 )
        {
          if ( *(_QWORD *)(v14 + 8) == 12 )
          {
            v15 = *(_QWORD **)v14;
            if ( **(_QWORD **)v14 == 0x636F6C6C61657270LL && *((_DWORD *)v15 + 2) == 1684370529 )
              break;
          }
          v14 += 56;
          if ( (unsigned __int64 *)v14 == v13 )
            goto LABEL_94;
        }
        v65 = **(_QWORD **)(v14 + 32);
        v16 = (__int64)v13 - v14 - 56;
        v17 = 0x6DB6DB6DB6DB6DB7LL * (v16 >> 3);
        if ( v16 <= 0 )
          goto LABEL_57;
        for ( i = (_QWORD *)(v14 + 72); ; v15 = (_QWORD *)*(i - 9) )
        {
          v24 = (_QWORD *)*(i - 2);
          if ( v24 == i )
          {
            v25 = *(i - 1);
            if ( v25 )
            {
              if ( v25 == 1 )
                *(_BYTE *)v15 = *(_BYTE *)i;
              else
                memcpy(v15, i, v25);
              v25 = *(i - 1);
              v15 = (_QWORD *)*(i - 9);
            }
            *(i - 8) = v25;
            *((_BYTE *)v15 + v25) = 0;
            v15 = (_QWORD *)*(i - 2);
          }
          else
          {
            if ( v15 == i - 7 )
            {
              *(i - 9) = v24;
              *(i - 8) = *(i - 1);
              *(i - 7) = *i;
            }
            else
            {
              *(i - 9) = v24;
              v19 = *(i - 7);
              *(i - 8) = *(i - 1);
              *(i - 7) = *i;
              if ( v15 )
              {
                *(i - 2) = v15;
                *i = v19;
                goto LABEL_23;
              }
            }
            *(i - 2) = i;
            v15 = i;
          }
LABEL_23:
          *(i - 1) = 0;
          *(_BYTE *)v15 = 0;
          v20 = i[2];
          v21 = *(i - 5);
          i[2] = 0;
          *(i - 5) = v20;
          v22 = i[3];
          i[3] = 0;
          *(i - 4) = v22;
          v23 = i[4];
          i[4] = 0;
          *(i - 3) = v23;
          if ( v21 )
            j_j___libc_free_0(v21);
          i += 7;
          if ( !--v17 )
            break;
        }
        v12 = v84;
        v11 = (__int64)v83;
LABEL_57:
        v29 = (unsigned int)(v12 - 1);
        LODWORD(v84) = v29;
        v30 = (unsigned __int64 *)(v11 + 56 * v29);
        v31 = v30[4];
        if ( v31 )
          j_j___libc_free_0(v31);
        if ( (unsigned __int64 *)*v30 != v30 + 2 )
          j_j___libc_free_0(*v30);
        v32 = *(_QWORD *)(v65 - 32LL * (*(_DWORD *)(v65 + 4) & 0x7FFFFFF));
        if ( *(_DWORD *)(v32 + 32) <= 0x40u )
          v33 = *(_QWORD *)(v32 + 24);
        else
          v33 = **(_QWORD **)(v32 + 24);
        v34 = (unsigned __int8 *)sub_B4BA60(v6, (__int64)v83, (unsigned int)v84, (__int64)(v6 + 24), 0);
        sub_BD84D0((__int64)v6, (__int64)v34);
        sub_BD6B90(v34, v6);
        sub_B43D60(v6);
        sub_D5F1F0((__int64)&v86, v65);
        v79 = 257;
        v35 = sub_AA4E30(v89);
        BYTE4(v75) = 0;
        v72 = (_BYTE *)sub_BCE3C0(v92, *(_DWORD *)(v35 + 4));
        v36 = sub_B33D10((__int64)&v86, 0x157u, (__int64)&v72, 1, 0, 0, (__int64)v75, (__int64)v78);
        v37 = sub_B46B10((__int64)v34, 0);
        sub_D5F1F0((__int64)&v86, v37);
        v79 = 257;
        BYTE4(v75) = 0;
        v69 = (_BYTE *)v36;
        v72 = *(_BYTE **)(v36 + 8);
        sub_B33D10((__int64)&v86, 0x156u, (__int64)&v72, 1, (int)&v69, 1, (__int64)v75, (__int64)v78);
        v40 = v71;
        v41 = v71;
        v69 = v71;
        v70 = 0x200000000LL;
        if ( v33 )
        {
          if ( v33 > 2 )
          {
            sub_C8D5F0((__int64)&v69, v71, v33, 8u, v38, v39);
            v41 = v69;
            v40 = &v69[8 * (unsigned int)v70];
          }
          for ( j = &v41[8 * v33]; j != v40; ++v40 )
          {
            if ( v40 )
              *v40 = 0;
          }
          LODWORD(v70) = v33;
        }
        v43 = *(_QWORD *)(v65 + 16);
        v72 = v74;
        v73 = 0x200000000LL;
        if ( v43 )
        {
          v44 = v43;
          v45 = 0;
          do
          {
            v44 = *(_QWORD *)(v44 + 8);
            ++v45;
          }
          while ( v44 );
          v46 = v74;
          if ( v45 > 2 )
          {
            sub_C8D5F0((__int64)&v72, v74, v45, 8u, v38, v39);
            v46 = &v72[8 * (unsigned int)v73];
          }
          do
          {
            v46 += 8;
            *((_QWORD *)v46 - 1) = *(_QWORD *)(v43 + 24);
            v43 = *(_QWORD *)(v43 + 8);
          }
          while ( v43 );
          v47 = (unsigned __int64)v72;
          LODWORD(v73) = v73 + v45;
          v67 = &v72[8 * (unsigned int)v73];
          if ( v72 != v67 )
          {
            do
            {
              v48 = *(_QWORD **)v47;
              v49 = *(_QWORD *)(*(_QWORD *)v47 + 32 * (1LL - (*(_DWORD *)(*(_QWORD *)v47 + 4LL) & 0x7FFFFFF)));
              v50 = *(_QWORD **)(v49 + 24);
              if ( *(_DWORD *)(v49 + 32) > 0x40u )
                v50 = (_QWORD *)*v50;
              v68 = 8LL * (_QWORD)v50;
              v51 = *(_QWORD *)&v69[8 * (_QWORD)v50];
              if ( !v51 )
              {
                v52 = v48[1];
                if ( (unsigned int)*(unsigned __int8 *)(v52 + 8) - 17 <= 1 )
                  v52 = **(_QWORD **)(v52 + 16);
                v53 = *(_DWORD *)(v52 + 8) >> 8;
                v78[0] = v48[9];
                v54 = sub_A747F0(v78, -1, 84);
                if ( !v54 )
                  v54 = sub_B495C0((__int64)v48, 84);
                v78[0] = v54;
                v55 = (__int64 *)sub_A72A60(v78);
                v56 = sub_B46B10(v65, 0);
                sub_D5F1F0((__int64)&v86, v56);
                v77 = 1;
                v75 = "paarg";
                v76 = 3;
                v57 = sub_AA4E30(v89);
                v64 = sub_AE5260(v57, (__int64)v55);
                v79 = 257;
                v58 = sub_BD2C40(80, unk_3F10A14);
                v51 = (__int64)v58;
                if ( v58 )
                  sub_B4CCA0((__int64)v58, v55, v53, 0, v64, (__int64)v78, 0, 0);
                (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v94 + 2))(
                  v94,
                  v51,
                  &v75,
                  v90,
                  v91);
                v59 = (unsigned __int64)v86;
                v60 = &v86[16 * (unsigned int)v87];
                if ( v86 != v60 )
                {
                  do
                  {
                    v61 = *(_QWORD *)(v59 + 8);
                    v62 = *(_DWORD *)v59;
                    v59 += 16LL;
                    sub_B99FD0(v51, v62, v61);
                  }
                  while ( v60 != (_BYTE *)v59 );
                }
                *(_QWORD *)&v69[v68] = v51;
              }
              v47 += 8LL;
              sub_BD84D0((__int64)v48, v51);
              sub_B43D60(v48);
            }
            while ( v67 != (_BYTE *)v47 );
          }
        }
        else
        {
          LODWORD(v73) = 0;
        }
        sub_B43D60((_QWORD *)v65);
        if ( v72 != v74 )
          _libc_free((unsigned __int64)v72);
        if ( v69 != v71 )
          _libc_free((unsigned __int64)v69);
        v26 = (__int64)v83;
        v27 = &v83[7 * (unsigned int)v84];
        if ( v83 != v27 )
        {
          do
          {
            v28 = *(v27 - 3);
            v27 -= 7;
            if ( v28 )
              j_j___libc_free_0(v28);
            if ( (unsigned __int64 *)*v27 != v27 + 2 )
              j_j___libc_free_0(*v27);
          }
          while ( (unsigned __int64 *)v26 != v27 );
          v27 = v83;
        }
        if ( v27 != (unsigned __int64 *)v85 )
          _libc_free((unsigned __int64)v27);
LABEL_47:
        if ( v63 == ++v66 )
        {
          v63 = v80;
          break;
        }
      }
    }
    if ( v63 != (unsigned __int8 **)v82 )
      _libc_free((unsigned __int64)v63);
  }
  nullsub_61();
  v101 = &unk_49DA100;
  nullsub_63();
  if ( v86 != v88 )
    _libc_free((unsigned __int64)v86);
}
