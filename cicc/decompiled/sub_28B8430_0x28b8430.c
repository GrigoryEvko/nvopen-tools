// Function: sub_28B8430
// Address: 0x28b8430
//
void __fastcall sub_28B8430(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r15
  unsigned int v4; // eax
  unsigned int v5; // eax
  unsigned int v6; // eax
  unsigned int v7; // ebx
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 *v10; // rcx
  __int64 *v11; // rax
  unsigned int v12; // eax
  char v13; // dl
  int v14; // esi
  __int64 *v15; // rsi
  __int64 v16; // r8
  __int64 v17; // rax
  int v18; // edi
  __int64 v19; // r12
  __int64 v20; // rbx
  int v21; // eax
  __int64 v22; // rax
  unsigned __int64 v23; // r14
  __int64 *v24; // r12
  __int64 v25; // r13
  __int64 v26; // r9
  __int64 v27; // r15
  __int64 v28; // rcx
  __int64 *v29; // rdi
  __int64 v30; // rsi
  __int64 *v31; // rax
  unsigned int v32; // eax
  int v33; // eax
  int v34; // edx
  __int64 *v35; // rdx
  __int64 v36; // r9
  __int64 v37; // rax
  int v38; // edi
  bool v39; // cc
  unsigned __int64 v40; // rdi
  int v41; // eax
  unsigned __int64 v42; // rdi
  int v43; // eax
  __int64 *v44; // rbx
  _QWORD *v45; // rax
  __int64 *v46; // rcx
  int v47; // esi
  int v48; // edx
  char v49; // r12
  __int64 *v50; // rsi
  __int64 *v51; // rax
  __int64 v52; // r8
  int v53; // edi
  __int64 i; // rax
  unsigned __int64 v55; // rdi
  unsigned __int64 v56; // rdi
  __int64 v57; // rcx
  __int64 v58; // rdx
  int v59; // eax
  __int64 *v60; // rdx
  __int64 *v61; // rax
  __int64 v62; // rsi
  __int64 *v63; // rax
  __int64 *v64; // rdx
  __int64 v65; // rsi
  unsigned int v66; // eax
  __int64 v67; // rcx
  unsigned int v68; // edx
  unsigned int v69; // eax
  __int64 v70; // rcx
  unsigned int v71; // edx
  __int64 v72; // [rsp+0h] [rbp-1A0h]
  __int64 v73; // [rsp+8h] [rbp-198h]
  __int64 v74; // [rsp+10h] [rbp-190h]
  _QWORD *v75; // [rsp+18h] [rbp-188h]
  unsigned int v77; // [rsp+3Ch] [rbp-164h]
  __int64 v78; // [rsp+40h] [rbp-160h]
  __int64 v79; // [rsp+48h] [rbp-158h]
  __int64 v80; // [rsp+50h] [rbp-150h]
  int v81; // [rsp+58h] [rbp-148h]
  char v82; // [rsp+5Fh] [rbp-141h]
  int v83; // [rsp+60h] [rbp-140h]
  int v84; // [rsp+64h] [rbp-13Ch]
  __int64 v85; // [rsp+68h] [rbp-138h]
  int v86; // [rsp+70h] [rbp-130h]
  int v87; // [rsp+74h] [rbp-12Ch]
  __int64 v88; // [rsp+78h] [rbp-128h]
  __int64 v89; // [rsp+80h] [rbp-120h]
  __int64 v90; // [rsp+88h] [rbp-118h]
  __int64 v91; // [rsp+90h] [rbp-110h]
  int v92; // [rsp+98h] [rbp-108h]
  unsigned int v93; // [rsp+9Ch] [rbp-104h]
  __int64 v94; // [rsp+A0h] [rbp-100h]
  __int64 v95; // [rsp+B0h] [rbp-F0h]
  __int64 v96; // [rsp+B8h] [rbp-E8h] BYREF
  __int64 v97; // [rsp+C0h] [rbp-E0h]
  __int64 v98; // [rsp+C8h] [rbp-D8h] BYREF
  unsigned int v99; // [rsp+D0h] [rbp-D0h]
  _BYTE v100[4]; // [rsp+108h] [rbp-98h] BYREF
  int v101; // [rsp+10Ch] [rbp-94h]
  __int64 v102; // [rsp+110h] [rbp-90h]
  __int64 v103; // [rsp+118h] [rbp-88h]
  unsigned int v104; // [rsp+120h] [rbp-80h]
  __int64 v105; // [rsp+128h] [rbp-78h]
  int v106; // [rsp+130h] [rbp-70h]
  __int64 v107; // [rsp+138h] [rbp-68h]
  __int64 v108; // [rsp+140h] [rbp-60h]
  int v109; // [rsp+148h] [rbp-58h]
  __int64 v110; // [rsp+150h] [rbp-50h]
  int v111; // [rsp+158h] [rbp-48h]
  int v112; // [rsp+160h] [rbp-40h]
  __int64 v113; // [rsp+168h] [rbp-38h]

  if ( a1 != a2 && a2 != a1 + 192 )
  {
    v2 = a1;
    v72 = a1 + 120;
    v3 = a1 + 216;
    v75 = (_QWORD *)(a1 + 24);
    do
    {
      v4 = *(_DWORD *)(v3 + 88);
      v7 = *(_DWORD *)(v2 + 112);
      v8 = v3 - 24;
      v93 = v4;
      if ( v4 == v7 )
        v4 = (unsigned int)sub_C4C880(v3 + 96, v72) >> 31;
      else
        LOBYTE(v4) = v4 < v7;
      if ( !(_BYTE)v4 )
      {
        if ( v93 == v7 )
          v5 = (unsigned int)sub_C4C880(v72, v3 + 96) >> 31;
        else
          LOBYTE(v5) = v93 > v7;
        if ( (_BYTE)v5
          || ((v6 = *(_DWORD *)(v2 + 152), *(_DWORD *)(v3 + 128) == v6)
            ? (v6 = (unsigned int)sub_C4C880(v3 + 136, v2 + 160) >> 31)
            : (LOBYTE(v6) = *(_DWORD *)(v3 + 128) < v6),
              !(_BYTE)v6) )
        {
          sub_28B5980((__int64 *)(v3 - 24));
          v91 = v3 + 168;
          goto LABEL_13;
        }
      }
      v9 = *(_QWORD *)(v3 - 24);
      v10 = (__int64 *)(v3 - 16);
      v96 = 0;
      v97 = 1;
      v95 = v9;
      v11 = &v98;
      do
        *v11++ = -4096;
      while ( v11 != (__int64 *)v100 );
      v12 = *(_DWORD *)(v8 + 16);
      v13 = v97;
      v77 = v12 >> 1;
      *(_DWORD *)(v8 + 16) = v97 & 0xFFFFFFFE | v12 & 1;
      v14 = *(_DWORD *)(v3 - 4);
      *(_DWORD *)(v3 - 4) = HIDWORD(v97);
      LODWORD(v97) = v12 & 0xFFFFFFFE | v13 & 1;
      v81 = v14;
      HIDWORD(v97) = v14;
      if ( (v13 & 1) != 0 )
      {
        v15 = &v96;
        if ( (*(_BYTE *)(v8 + 16) & 1) != 0 )
        {
          v63 = &v98;
          v64 = (__int64 *)v3;
          do
          {
            v65 = *v63;
            *v63++ = *v64;
            *v64++ = v65;
          }
          while ( v63 != (__int64 *)v100 );
          goto LABEL_23;
        }
      }
      else
      {
        v15 = (__int64 *)(v3 - 16);
        v10 = &v96;
        if ( (*(_BYTE *)(v8 + 16) & 1) == 0 )
        {
          v66 = v99;
          v67 = *(_QWORD *)v3;
          *(_QWORD *)v3 = v98;
          v68 = *(_DWORD *)(v3 + 8);
          v98 = v67;
          v99 = v68;
          *(_DWORD *)(v3 + 8) = v66;
          goto LABEL_24;
        }
      }
      *((_BYTE *)v10 + 8) |= 1u;
      v16 = v10[2];
      v17 = 2;
      v18 = *((_DWORD *)v10 + 6);
      do
      {
        v10[v17] = v15[v17];
        ++v17;
      }
      while ( v17 != 10 );
      *((_BYTE *)v15 + 8) &= ~1u;
      v15[2] = v16;
      *((_DWORD *)v15 + 6) = v18;
LABEL_23:
      v93 = *(_DWORD *)(v3 + 88);
      v77 = (unsigned int)v97 >> 1;
      v81 = HIDWORD(v97);
LABEL_24:
      v19 = v8 - v2;
      v20 = v3 + 168;
      v82 = *(_BYTE *)(v3 + 64);
      v100[0] = v82;
      v92 = *(_DWORD *)(v3 + 68);
      v101 = v92;
      v90 = *(_QWORD *)(v3 + 72);
      v102 = v90;
      v80 = *(_QWORD *)(v3 + 80);
      v103 = v80;
      v104 = v93;
      v21 = *(_DWORD *)(v3 + 104);
      *(_DWORD *)(v3 + 104) = 0;
      v84 = v21;
      v106 = v21;
      v79 = *(_QWORD *)(v3 + 96);
      v105 = v79;
      v85 = *(_QWORD *)(v3 + 112);
      v107 = v85;
      v89 = *(_QWORD *)(v3 + 120);
      v108 = v89;
      v87 = *(_DWORD *)(v3 + 128);
      v109 = v87;
      v83 = *(_DWORD *)(v3 + 144);
      v111 = v83;
      v22 = *(_QWORD *)(v3 + 136);
      *(_DWORD *)(v3 + 144) = 0;
      v88 = v22;
      v110 = v22;
      v91 = v3 + 168;
      v86 = *(_DWORD *)(v3 + 152);
      v112 = v86;
      v78 = *(_QWORD *)(v3 + 160);
      v113 = v78;
      v23 = 0xAAAAAAAAAAAAAAABLL * (v19 >> 6);
      if ( v19 <= 0 )
        goto LABEL_46;
      v74 = v3;
      v24 = (__int64 *)(v3 + 64);
      v73 = v2;
      v25 = v3 - 16;
      do
      {
        v26 = v20 - 384;
        v20 -= 192;
        v27 = v25;
        v25 = v20 - 184;
        *(_QWORD *)v20 = *(_QWORD *)(v20 - 192);
        v28 = v20 - 184;
        if ( (*(_BYTE *)(v20 + 16) & 1) == 0 )
        {
          v94 = v26;
          sub_C7D6A0(*(_QWORD *)(v20 + 24), 8LL * *(unsigned int *)(v20 + 32), 8);
          v26 = v94;
          v28 = v20 - 184;
        }
        *(_DWORD *)(v20 + 16) = 1;
        v29 = (__int64 *)(v20 + 24);
        *(_DWORD *)(v20 + 20) = 0;
        v30 = v20 + 24;
        v31 = (__int64 *)(v20 + 24);
        do
        {
          if ( v31 )
            *v31 = -4096;
          ++v31;
        }
        while ( v31 != v24 );
        v32 = *(_DWORD *)(v26 + 16) & 0xFFFFFFFE;
        *(_DWORD *)(v26 + 16) = *(_DWORD *)(v20 + 16) & 0xFFFFFFFE | *(_DWORD *)(v26 + 16) & 1;
        v33 = v32 | *(_DWORD *)(v20 + 16) & 1;
        v34 = *(_DWORD *)(v20 + 20);
        *(_DWORD *)(v20 + 20) = *(_DWORD *)(v20 - 172);
        *(_DWORD *)(v20 + 16) = v33;
        *(_DWORD *)(v20 - 172) = v34;
        if ( (v33 & 1) == 0 )
        {
          if ( (*(_BYTE *)(v26 + 16) & 1) == 0 )
          {
            v58 = *(_QWORD *)(v20 - 168);
            *(_QWORD *)(v20 - 168) = *(_QWORD *)(v20 + 24);
            v59 = *(_DWORD *)(v20 + 32);
            *(_QWORD *)(v20 + 24) = v58;
            LODWORD(v58) = *(_DWORD *)(v20 - 160);
            *(_DWORD *)(v20 - 160) = v59;
            *(_DWORD *)(v20 + 32) = v58;
            goto LABEL_38;
          }
          v28 = v27;
          v35 = (__int64 *)(v20 + 24);
          v30 = v20 - 168;
          v27 = v20 - 184;
LABEL_35:
          *(_BYTE *)(v28 + 8) |= 1u;
          v36 = *(_QWORD *)(v28 + 16);
          v37 = 0;
          v38 = *(_DWORD *)(v28 + 24);
          do
          {
            v35[v37] = *(_QWORD *)(v30 + v37 * 8);
            ++v37;
          }
          while ( v37 != 8 );
          *(_BYTE *)(v27 + 8) &= ~1u;
          *(_QWORD *)(v27 + 16) = v36;
          *(_DWORD *)(v27 + 24) = v38;
          goto LABEL_38;
        }
        v35 = (__int64 *)(v20 - 168);
        if ( (*(_BYTE *)(v26 + 16) & 1) == 0 )
          goto LABEL_35;
        do
        {
          v57 = *v29;
          *v29++ = *v35;
          *v35++ = v57;
        }
        while ( v24 != v29 );
LABEL_38:
        v39 = *(_DWORD *)(v20 + 128) <= 0x40u;
        *(_BYTE *)(v20 + 88) = *(_BYTE *)(v20 - 104);
        *(_DWORD *)(v20 + 92) = *(_DWORD *)(v20 - 100);
        *(_QWORD *)(v20 + 96) = *(_QWORD *)(v20 - 96);
        *(_QWORD *)(v20 + 104) = *(_QWORD *)(v20 - 88);
        *(_DWORD *)(v20 + 112) = *(_DWORD *)(v20 - 80);
        if ( !v39 )
        {
          v40 = *(_QWORD *)(v20 + 120);
          if ( v40 )
            j_j___libc_free_0_0(v40);
        }
        v39 = *(_DWORD *)(v20 + 168) <= 0x40u;
        *(_QWORD *)(v20 + 120) = *(_QWORD *)(v20 - 72);
        v41 = *(_DWORD *)(v20 - 64);
        *(_DWORD *)(v20 - 64) = 0;
        *(_DWORD *)(v20 + 128) = v41;
        *(_QWORD *)(v20 + 136) = *(_QWORD *)(v20 - 56);
        *(_QWORD *)(v20 + 144) = *(_QWORD *)(v20 - 48);
        *(_DWORD *)(v20 + 152) = *(_DWORD *)(v20 - 40);
        if ( !v39 )
        {
          v42 = *(_QWORD *)(v20 + 160);
          if ( v42 )
            j_j___libc_free_0_0(v42);
        }
        v24 -= 24;
        *(_QWORD *)(v20 + 160) = *(_QWORD *)(v20 - 32);
        v43 = *(_DWORD *)(v20 - 24);
        *(_DWORD *)(v20 - 24) = 0;
        *(_DWORD *)(v20 + 168) = v43;
        *(_DWORD *)(v20 + 176) = *(_DWORD *)(v20 - 16);
        *(_QWORD *)(v20 + 184) = *(_QWORD *)(v20 - 8);
        --v23;
      }
      while ( v23 );
      v3 = v74;
      v2 = v73;
LABEL_46:
      v44 = (__int64 *)(v2 + 8);
      *(_QWORD *)v2 = v95;
      if ( (*(_BYTE *)(v2 + 16) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v2 + 24), 8LL * *(unsigned int *)(v2 + 32), 8);
      v45 = v75;
      *(_QWORD *)(v2 + 16) = 1;
      v46 = v75;
      do
      {
        if ( v45 )
          *v45 = -4096;
        ++v45;
      }
      while ( v45 != (_QWORD *)(v2 + 88) );
      v47 = *(_DWORD *)(v2 + 16);
      LODWORD(v97) = v47 & 0xFFFFFFFE | v97 & 1;
      v48 = *(_DWORD *)(v2 + 20);
      *(_DWORD *)(v2 + 20) = v81;
      *(_DWORD *)(v2 + 16) = (2 * v77) | v47 & 1;
      HIDWORD(v97) = v48;
      if ( (v47 & 1) != 0 )
      {
        v50 = &v98;
        v51 = &v96;
        if ( (v97 & 1) == 0 )
          goto LABEL_55;
        v60 = v75;
        v61 = &v98;
        do
        {
          v62 = *v60;
          *v60++ = *v61++;
          *(v61 - 1) = v62;
        }
        while ( v100 != (_BYTE *)v61 );
LABEL_58:
        v82 = v100[0];
        v49 = v97 & 1;
        v92 = v101;
        v86 = v112;
        v78 = v113;
        v90 = v102;
        v80 = v103;
        v93 = v104;
        v79 = v105;
        v84 = v106;
        v85 = v107;
        v89 = v108;
        v87 = v109;
        v88 = v110;
        v83 = v111;
      }
      else
      {
        v49 = v97 & 1;
        if ( (v97 & 1) != 0 )
        {
          v50 = v75;
          v46 = &v98;
          v51 = (__int64 *)(v2 + 8);
          v44 = &v96;
LABEL_55:
          *((_BYTE *)v51 + 8) |= 1u;
          v52 = v51[2];
          v53 = *((_DWORD *)v51 + 6);
          for ( i = 0; i != 8; ++i )
            v50[i] = v46[i];
          *((_BYTE *)v44 + 8) &= ~1u;
          v44[2] = v52;
          *((_DWORD *)v44 + 6) = v53;
          goto LABEL_58;
        }
        v69 = *(_DWORD *)(v2 + 32);
        v70 = v98;
        v98 = *(_QWORD *)(v2 + 24);
        v71 = v99;
        *(_QWORD *)(v2 + 24) = v70;
        *(_DWORD *)(v2 + 32) = v71;
        v99 = v69;
      }
      v39 = *(_DWORD *)(v2 + 128) <= 0x40u;
      *(_BYTE *)(v2 + 88) = v82;
      *(_DWORD *)(v2 + 92) = v92;
      *(_QWORD *)(v2 + 96) = v90;
      *(_QWORD *)(v2 + 104) = v80;
      *(_DWORD *)(v2 + 112) = v93;
      if ( !v39 )
      {
        v55 = *(_QWORD *)(v2 + 120);
        if ( v55 )
          j_j___libc_free_0_0(v55);
      }
      v39 = *(_DWORD *)(v2 + 168) <= 0x40u;
      *(_QWORD *)(v2 + 120) = v79;
      *(_DWORD *)(v2 + 128) = v84;
      *(_QWORD *)(v2 + 136) = v85;
      *(_QWORD *)(v2 + 144) = v89;
      *(_DWORD *)(v2 + 152) = v87;
      if ( !v39 )
      {
        v56 = *(_QWORD *)(v2 + 160);
        if ( v56 )
          j_j___libc_free_0_0(v56);
      }
      *(_QWORD *)(v2 + 160) = v88;
      *(_DWORD *)(v2 + 168) = v83;
      *(_DWORD *)(v2 + 176) = v86;
      *(_QWORD *)(v2 + 184) = v78;
      if ( !v49 )
        sub_C7D6A0(v98, 8LL * v99, 8);
LABEL_13:
      v3 += 192;
    }
    while ( a2 != v91 );
  }
}
