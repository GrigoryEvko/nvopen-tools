// Function: sub_1D865C0
// Address: 0x1d865c0
//
void __fastcall sub_1D865C0(__int64 a1)
{
  bool v2; // zf
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rsi
  unsigned __int8 *v8; // rsi
  _BYTE *v9; // rdx
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // r15
  __int64 v18; // r14
  _QWORD *v19; // rax
  _QWORD *v20; // r13
  __int64 v21; // rdi
  unsigned __int64 *v22; // r14
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned __int8 *v29; // rsi
  __int64 v30; // rsi
  __int64 v31; // r12
  __int64 v32; // rbx
  int v33; // eax
  __int64 v34; // rax
  int v35; // edx
  __int64 v36; // rdx
  _QWORD *v37; // rax
  __int64 v38; // rcx
  unsigned __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rcx
  _QWORD *v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rdx
  unsigned __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdi
  unsigned __int64 *v55; // r13
  __int64 v56; // rax
  unsigned __int64 v57; // rcx
  __int64 v58; // rsi
  __int64 v59; // rsi
  unsigned __int8 *v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rsi
  __int64 v64; // rsi
  unsigned __int8 *v65; // rsi
  _QWORD *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // r13
  __int64 v74; // r15
  __int64 v75; // r14
  int v76; // eax
  __int64 v77; // rax
  int v78; // edx
  __int64 v79; // rdx
  __int64 *v80; // rax
  __int64 v81; // rcx
  unsigned __int64 v82; // rdx
  __int64 v83; // rdx
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r14
  _QWORD *v87; // rax
  _QWORD *v88; // r13
  __int64 v89; // rdi
  unsigned __int64 *v90; // r14
  __int64 v91; // rax
  unsigned __int64 v92; // rcx
  __int64 v93; // rsi
  __int64 v94; // rsi
  unsigned __int8 *v95; // rsi
  _QWORD *v96; // [rsp+8h] [rbp-88h]
  __int64 *v97; // [rsp+10h] [rbp-80h]
  __int64 *v98; // [rsp+18h] [rbp-78h]
  __int64 v99; // [rsp+18h] [rbp-78h]
  __int64 v100[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v101; // [rsp+30h] [rbp-60h]
  __int64 v102[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v103; // [rsp+50h] [rbp-40h]

  v2 = *(_BYTE *)(a1 + 104) == 0;
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = sub_157EE30(v3);
    if ( !v4 )
    {
      *(_QWORD *)(a1 + 136) = 0;
      *(_QWORD *)(a1 + 128) = *(_QWORD *)(a1 + 8);
      BUG();
    }
    v5 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 136) = v4;
    *(_QWORD *)(a1 + 128) = v5;
    if ( v4 == v5 + 40 )
      goto LABEL_9;
    v6 = *(_QWORD *)(v4 + 24);
    v102[0] = v6;
    if ( v6 )
    {
      sub_1623A60((__int64)v102, v6, 2);
      v7 = *(_QWORD *)(a1 + 120);
      if ( !v7 )
        goto LABEL_7;
    }
    else
    {
      v7 = *(_QWORD *)(a1 + 120);
      if ( !v7 )
      {
LABEL_9:
        v9 = *(_BYTE **)(a1 + 16);
        v103 = 257;
        v10 = sub_12AA0C0((__int64 *)(a1 + 120), 0x24u, v9, *(_QWORD *)(a1 + 24), (__int64)v102);
        v11 = *(_QWORD **)(a1 + 144);
        v101 = 257;
        v12 = v10;
        v13 = sub_1643350(v11);
        v14 = sub_159C470(v13, 1, 0);
        v15 = sub_1643350(*(_QWORD **)(a1 + 144));
        v16 = sub_159C470(v15, -1, 0);
        if ( *(_BYTE *)(v12 + 16) > 0x10u || *(_BYTE *)(v16 + 16) > 0x10u || *(_BYTE *)(v14 + 16) > 0x10u )
        {
          v98 = (__int64 *)v16;
          v103 = 257;
          v43 = sub_1648A60(56, 3u);
          v17 = v43;
          if ( v43 )
          {
            v44 = *v98;
            v96 = v43 - 9;
            v97 = v98;
            v99 = (__int64)v43;
            sub_15F1EA0((__int64)v43, v44, 55, (__int64)(v43 - 9), 3, 0);
            if ( *(v17 - 9) )
            {
              v45 = *(v17 - 8);
              v46 = *(v17 - 7) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v46 = v45;
              if ( v45 )
                *(_QWORD *)(v45 + 16) = *(_QWORD *)(v45 + 16) & 3LL | v46;
            }
            *(v17 - 9) = v12;
            v47 = *(_QWORD *)(v12 + 8);
            *(v17 - 8) = v47;
            if ( v47 )
              *(_QWORD *)(v47 + 16) = (unsigned __int64)(v17 - 8) | *(_QWORD *)(v47 + 16) & 3LL;
            *(v17 - 7) = (v12 + 8) | *(v17 - 7) & 3LL;
            *(_QWORD *)(v12 + 8) = v96;
            if ( *(v17 - 6) )
            {
              v48 = *(v17 - 5);
              v49 = *(v17 - 4) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v49 = v48;
              if ( v48 )
                *(_QWORD *)(v48 + 16) = *(_QWORD *)(v48 + 16) & 3LL | v49;
            }
            *(v17 - 6) = v97;
            v50 = v97[1];
            *(v17 - 5) = v50;
            if ( v50 )
              *(_QWORD *)(v50 + 16) = (unsigned __int64)(v17 - 5) | *(_QWORD *)(v50 + 16) & 3LL;
            *(v17 - 4) = (unsigned __int64)(v97 + 1) | *(v17 - 4) & 3LL;
            v97[1] = (__int64)(v17 - 6);
            if ( *(v17 - 3) )
            {
              v51 = *(v17 - 2);
              v52 = *(v17 - 1) & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v52 = v51;
              if ( v51 )
                *(_QWORD *)(v51 + 16) = *(_QWORD *)(v51 + 16) & 3LL | v52;
            }
            *(v17 - 3) = v14;
            if ( v14 )
            {
              v53 = *(_QWORD *)(v14 + 8);
              *(v17 - 2) = v53;
              if ( v53 )
                *(_QWORD *)(v53 + 16) = (unsigned __int64)(v17 - 2) | *(_QWORD *)(v53 + 16) & 3LL;
              *(v17 - 1) = (v14 + 8) | *(v17 - 1) & 3LL;
              *(_QWORD *)(v14 + 8) = v17 - 3;
            }
            sub_164B780((__int64)v17, v102);
          }
          else
          {
            v99 = 0;
          }
          v54 = *(_QWORD *)(a1 + 128);
          if ( v54 )
          {
            v55 = *(unsigned __int64 **)(a1 + 136);
            sub_157E9D0(v54 + 40, (__int64)v17);
            v56 = v17[3];
            v57 = *v55;
            v17[4] = v55;
            v57 &= 0xFFFFFFFFFFFFFFF8LL;
            v17[3] = v57 | v56 & 7;
            *(_QWORD *)(v57 + 8) = v17 + 3;
            *v55 = *v55 & 7 | (unsigned __int64)(v17 + 3);
          }
          sub_164B780(v99, v100);
          v58 = *(_QWORD *)(a1 + 120);
          if ( v58 )
          {
            v102[0] = *(_QWORD *)(a1 + 120);
            sub_1623A60((__int64)v102, v58, 2);
            v59 = v17[6];
            if ( v59 )
              sub_161E7C0((__int64)(v17 + 6), v59);
            v60 = (unsigned __int8 *)v102[0];
            v17[6] = v102[0];
            if ( v60 )
              sub_1623210((__int64)v102, v60, (__int64)(v17 + 6));
          }
        }
        else
        {
          v17 = (_QWORD *)sub_15A2DC0(v12, (__int64 *)v16, v14, 0);
        }
        v18 = *(_QWORD *)(a1 + 88);
        v19 = sub_1648A60(56, 1u);
        v20 = v19;
        if ( v19 )
          sub_15F8320((__int64)v19, v18, 0);
        v21 = *(_QWORD *)(a1 + 128);
        v103 = 257;
        if ( v21 )
        {
          v22 = *(unsigned __int64 **)(a1 + 136);
          sub_157E9D0(v21 + 40, (__int64)v20);
          v23 = v20[3];
          v24 = *v22;
          v20[4] = v22;
          v24 &= 0xFFFFFFFFFFFFFFF8LL;
          v20[3] = v24 | v23 & 7;
          *(_QWORD *)(v24 + 8) = v20 + 3;
          *v22 = *v22 & 7 | (unsigned __int64)(v20 + 3);
        }
        sub_164B780((__int64)v20, v102);
        v29 = *(unsigned __int8 **)(a1 + 120);
        if ( v29 )
        {
          v100[0] = *(_QWORD *)(a1 + 120);
          sub_1623A60((__int64)v100, (__int64)v29, 2);
          v30 = v20[6];
          if ( v30 )
            sub_161E7C0((__int64)(v20 + 6), v30);
          v29 = (unsigned __int8 *)v100[0];
          v20[6] = v100[0];
          if ( v29 )
            sub_1623210((__int64)v100, v29, (__int64)(v20 + 6));
        }
        v31 = *(_QWORD *)(a1 + 96);
        v32 = *(_QWORD *)(a1 + 8);
        v33 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        if ( v33 == *(_DWORD *)(v31 + 56) )
        {
          sub_15F55D0(v31, (__int64)v29, v25, v26, v27, v28);
          v33 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        }
        v34 = (v33 + 1) & 0xFFFFFFF;
        v35 = v34 | *(_DWORD *)(v31 + 20) & 0xF0000000;
        *(_DWORD *)(v31 + 20) = v35;
        if ( (v35 & 0x40000000) != 0 )
          v36 = *(_QWORD *)(v31 - 8);
        else
          v36 = v31 - 24 * v34;
        v37 = (_QWORD *)(v36 + 24LL * (unsigned int)(v34 - 1));
        if ( *v37 )
        {
          v38 = v37[1];
          v39 = v37[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v39 = v38;
          if ( v38 )
            *(_QWORD *)(v38 + 16) = *(_QWORD *)(v38 + 16) & 3LL | v39;
        }
        *v37 = v17;
        if ( v17 )
        {
          v40 = v17[1];
          v37[1] = v40;
          if ( v40 )
            *(_QWORD *)(v40 + 16) = (unsigned __int64)(v37 + 1) | *(_QWORD *)(v40 + 16) & 3LL;
          v37[2] = (unsigned __int64)(v17 + 1) | v37[2] & 3LL;
          v17[1] = v37;
        }
        v41 = *(_DWORD *)(v31 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v31 + 23) & 0x40) != 0 )
          v42 = *(_QWORD *)(v31 - 8);
        else
          v42 = v31 - 24 * v41;
        *(_QWORD *)(v42 + 8LL * (unsigned int)(v41 - 1) + 24LL * *(unsigned int *)(v31 + 56) + 8) = v32;
        return;
      }
    }
    sub_161E7C0(a1 + 120, v7);
LABEL_7:
    v8 = (unsigned __int8 *)v102[0];
    *(_QWORD *)(a1 + 120) = v102[0];
    if ( v8 )
      sub_1623210((__int64)v102, v8, a1 + 120);
    goto LABEL_9;
  }
  v61 = sub_157EE30(v3);
  if ( !v61 )
  {
    *(_QWORD *)(a1 + 136) = 0;
    *(_QWORD *)(a1 + 128) = *(_QWORD *)(a1 + 8);
    BUG();
  }
  v62 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 136) = v61;
  *(_QWORD *)(a1 + 128) = v62;
  if ( v61 != v62 + 40 )
  {
    v63 = *(_QWORD *)(v61 + 24);
    v102[0] = v63;
    if ( v63 )
    {
      sub_1623A60((__int64)v102, v63, 2);
      v64 = *(_QWORD *)(a1 + 120);
      if ( !v64 )
        goto LABEL_68;
      goto LABEL_67;
    }
    v64 = *(_QWORD *)(a1 + 120);
    if ( v64 )
    {
LABEL_67:
      sub_161E7C0(a1 + 120, v64);
LABEL_68:
      v65 = (unsigned __int8 *)v102[0];
      *(_QWORD *)(a1 + 120) = v102[0];
      if ( v65 )
        sub_1623210((__int64)v102, v65, a1 + 120);
    }
  }
  v66 = (_QWORD *)sub_16498A0(*(_QWORD *)a1);
  v67 = sub_1643350(v66);
  v68 = sub_159C470(v67, 1, 0);
  v73 = *(_QWORD *)(a1 + 96);
  v74 = *(_QWORD *)(a1 + 8);
  v75 = v68;
  v76 = *(_DWORD *)(v73 + 20) & 0xFFFFFFF;
  if ( v76 == *(_DWORD *)(v73 + 56) )
  {
    sub_15F55D0(*(_QWORD *)(a1 + 96), 1, v69, v70, v71, v72);
    v76 = *(_DWORD *)(v73 + 20) & 0xFFFFFFF;
  }
  v77 = (v76 + 1) & 0xFFFFFFF;
  v78 = v77 | *(_DWORD *)(v73 + 20) & 0xF0000000;
  *(_DWORD *)(v73 + 20) = v78;
  if ( (v78 & 0x40000000) != 0 )
    v79 = *(_QWORD *)(v73 - 8);
  else
    v79 = v73 - 24 * v77;
  v80 = (__int64 *)(v79 + 24LL * (unsigned int)(v77 - 1));
  if ( *v80 )
  {
    v81 = v80[1];
    v82 = v80[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v82 = v81;
    if ( v81 )
      *(_QWORD *)(v81 + 16) = *(_QWORD *)(v81 + 16) & 3LL | v82;
  }
  *v80 = v75;
  if ( v75 )
  {
    v83 = *(_QWORD *)(v75 + 8);
    v80[1] = v83;
    if ( v83 )
      *(_QWORD *)(v83 + 16) = (unsigned __int64)(v80 + 1) | *(_QWORD *)(v83 + 16) & 3LL;
    v80[2] = (v75 + 8) | v80[2] & 3;
    *(_QWORD *)(v75 + 8) = v80;
  }
  v84 = *(_DWORD *)(v73 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v73 + 23) & 0x40) != 0 )
    v85 = *(_QWORD *)(v73 - 8);
  else
    v85 = v73 - 24 * v84;
  *(_QWORD *)(v85 + 8LL * (unsigned int)(v84 - 1) + 24LL * *(unsigned int *)(v73 + 56) + 8) = v74;
  v86 = *(_QWORD *)(a1 + 88);
  v87 = sub_1648A60(56, 1u);
  v88 = v87;
  if ( v87 )
    sub_15F8320((__int64)v87, v86, 0);
  v103 = 257;
  v89 = *(_QWORD *)(a1 + 128);
  if ( v89 )
  {
    v90 = *(unsigned __int64 **)(a1 + 136);
    sub_157E9D0(v89 + 40, (__int64)v88);
    v91 = v88[3];
    v92 = *v90;
    v88[4] = v90;
    v92 &= 0xFFFFFFFFFFFFFFF8LL;
    v88[3] = v92 | v91 & 7;
    *(_QWORD *)(v92 + 8) = v88 + 3;
    *v90 = *v90 & 7 | (unsigned __int64)(v88 + 3);
  }
  sub_164B780((__int64)v88, v102);
  v93 = *(_QWORD *)(a1 + 120);
  if ( v93 )
  {
    v100[0] = *(_QWORD *)(a1 + 120);
    sub_1623A60((__int64)v100, v93, 2);
    v94 = v88[6];
    if ( v94 )
      sub_161E7C0((__int64)(v88 + 6), v94);
    v95 = (unsigned __int8 *)v100[0];
    v88[6] = v100[0];
    if ( v95 )
      sub_1623210((__int64)v100, v95, (__int64)(v88 + 6));
  }
}
