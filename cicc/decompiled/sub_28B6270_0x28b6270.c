// Function: sub_28B6270
// Address: 0x28b6270
//
void __fastcall sub_28B6270(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // r9
  __int64 v7; // r13
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rbx
  unsigned int v12; // r10d
  unsigned int v13; // r11d
  unsigned int v14; // eax
  unsigned int v15; // eax
  unsigned int v16; // eax
  __int64 v17; // rbx
  __int64 v18; // r9
  __int64 v19; // rbx
  __int64 *v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // rsi
  __int64 *v23; // rdi
  unsigned int v24; // edi
  int v25; // edi
  __int64 v26; // r10
  __int64 *v27; // rdi
  __int64 v28; // rdx
  int v29; // eax
  __int64 i; // r10
  __int64 v31; // rax
  bool v32; // cc
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  __int64 v35; // rsi
  unsigned int v36; // eax
  unsigned int v37; // eax
  __int64 v38; // rax
  int v39; // edx
  __int64 v40; // r15
  unsigned int v41; // eax
  __int64 v42; // rdi
  __int64 v43; // r9
  __int64 v44; // rbx
  __int64 v45; // r14
  __int64 *v46; // rdx
  __int64 *v47; // rax
  __int64 v48; // rcx
  __int64 *v49; // rsi
  unsigned int v50; // esi
  int v51; // esi
  __int64 v52; // rax
  __int64 *v53; // rsi
  __int64 v54; // rdi
  __int64 v55; // rax
  int v56; // edx
  unsigned __int64 v57; // rdi
  unsigned __int64 v58; // rdi
  __int64 v59; // rdi
  __int64 v60; // rax
  int v61; // edx
  unsigned int v62; // eax
  __int64 v64; // [rsp+10h] [rbp-140h]
  __int64 v66; // [rsp+20h] [rbp-130h]
  unsigned int v67; // [rsp+2Ch] [rbp-124h]
  unsigned int v68; // [rsp+30h] [rbp-120h]
  __int64 v69; // [rsp+30h] [rbp-120h]
  __int64 v70; // [rsp+30h] [rbp-120h]
  __int64 v71; // [rsp+38h] [rbp-118h]
  __int64 v72; // [rsp+38h] [rbp-118h]
  __int64 v73; // [rsp+38h] [rbp-118h]
  __int64 v74; // [rsp+40h] [rbp-110h]
  __int64 v76; // [rsp+50h] [rbp-100h]
  __int64 v77; // [rsp+50h] [rbp-100h]
  __int64 v78; // [rsp+50h] [rbp-100h]
  __int64 v79; // [rsp+58h] [rbp-F8h]
  __int64 v80; // [rsp+58h] [rbp-F8h]
  _BYTE v81[24]; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v82; // [rsp+78h] [rbp-D8h]
  unsigned int v83; // [rsp+80h] [rbp-D0h]
  unsigned int v84; // [rsp+D0h] [rbp-80h]
  unsigned __int64 v85; // [rsp+D8h] [rbp-78h] BYREF
  unsigned int v86; // [rsp+E0h] [rbp-70h]
  unsigned int v87; // [rsp+F8h] [rbp-58h]
  unsigned __int64 v88; // [rsp+100h] [rbp-50h] BYREF
  unsigned int v89; // [rsp+108h] [rbp-48h]

  v4 = a1;
  v5 = a1 + 192 * a2;
  v64 = a3 & 1;
  v74 = (a3 - 1) / 2;
  if ( a2 < v74 )
  {
    v6 = a2;
    v76 = v5 + 96;
    v7 = v5 + 8;
    v79 = v5 + 136;
    while ( 1 )
    {
      v9 = 2 * (v6 + 1);
      v10 = v9 - 1;
      v11 = a1 + 384 * (v6 + 1);
      v5 = a1 + 192 * (v9 - 1);
      v12 = *(_DWORD *)(v11 + 112);
      v13 = *(_DWORD *)(v5 + 112);
      if ( v12 == v13 )
      {
        v66 = v6;
        v67 = *(_DWORD *)(v5 + 112);
        v68 = *(_DWORD *)(v11 + 112);
        v71 = 2 * (v6 + 1);
        v36 = sub_C4C880(v11 + 120, v5 + 120);
        v9 = v71;
        v12 = v68;
        v13 = v67;
        v6 = v66;
        v14 = v36 >> 31;
      }
      else
      {
        LOBYTE(v14) = v12 < v13;
      }
      if ( !(_BYTE)v14 )
      {
        if ( v12 == v13 )
        {
          v69 = v6;
          v72 = v9;
          v37 = sub_C4C880(v5 + 120, v11 + 120);
          v9 = v72;
          v6 = v69;
          v15 = v37 >> 31;
        }
        else
        {
          LOBYTE(v15) = v12 > v13;
        }
        if ( (_BYTE)v15 )
        {
          v5 = v11;
          v10 = v9;
        }
        else
        {
          v16 = *(_DWORD *)(v5 + 152);
          if ( *(_DWORD *)(v11 + 152) == v16 )
          {
            v70 = v6;
            v73 = v9;
            v62 = sub_C4C880(v11 + 160, v5 + 160);
            v9 = v73;
            v6 = v70;
            v16 = v62 >> 31;
          }
          else
          {
            LOBYTE(v16) = *(_DWORD *)(v11 + 152) < v16;
          }
          if ( !(_BYTE)v16 )
          {
            v5 = v11;
            v10 = v9;
          }
        }
      }
      v17 = 192 * v6;
      v18 = v5 + 8;
      v19 = a1 + v17;
      *(_QWORD *)v19 = *(_QWORD *)v5;
      if ( (*(_BYTE *)(v19 + 16) & 1) == 0 )
      {
        sub_C7D6A0(*(_QWORD *)(v19 + 24), 8LL * *(unsigned int *)(v19 + 32), 8);
        v18 = v5 + 8;
      }
      *(_QWORD *)(v19 + 16) = 1;
      v20 = (__int64 *)(v19 + 24);
      v21 = (__int64 *)(v19 + 88);
      v22 = v19 + 24;
      v23 = (__int64 *)(v19 + 24);
      do
      {
        if ( v23 )
          *v23 = -4096;
        ++v23;
      }
      while ( v23 != v21 );
      v24 = *(_DWORD *)(v5 + 16) & 0xFFFFFFFE;
      *(_DWORD *)(v5 + 16) = *(_DWORD *)(v19 + 16) & 0xFFFFFFFE | *(_DWORD *)(v5 + 16) & 1;
      *(_DWORD *)(v19 + 16) = v24 | *(_DWORD *)(v19 + 16) & 1;
      v25 = *(_DWORD *)(v19 + 20);
      *(_DWORD *)(v19 + 20) = *(_DWORD *)(v5 + 20);
      *(_DWORD *)(v5 + 20) = v25;
      if ( (*(_BYTE *)(v19 + 16) & 1) == 0 )
        break;
      v27 = (__int64 *)(v5 + 24);
      v26 = v18;
      if ( (*(_BYTE *)(v5 + 16) & 1) == 0 )
        goto LABEL_22;
      do
      {
        v35 = *v20;
        *v20++ = *v27;
        *v27++ = v35;
      }
      while ( v21 != v20 );
LABEL_25:
      *(_BYTE *)(v19 + 88) = *(_BYTE *)(v5 + 88);
      *(_DWORD *)(v19 + 92) = *(_DWORD *)(v5 + 92);
      v31 = v5 + 96;
      if ( v5 + 96 != v76 )
      {
        v32 = *(_DWORD *)(v19 + 128) <= 0x40u;
        *(_QWORD *)(v19 + 96) = *(_QWORD *)(v5 + 96);
        *(_QWORD *)(v19 + 104) = *(_QWORD *)(v5 + 104);
        *(_DWORD *)(v19 + 112) = *(_DWORD *)(v5 + 112);
        if ( !v32 )
        {
          v33 = *(_QWORD *)(v19 + 120);
          if ( v33 )
          {
            v77 = v18;
            j_j___libc_free_0_0(v33);
            v31 = v5 + 96;
            v18 = v77;
          }
        }
        *(_QWORD *)(v19 + 120) = *(_QWORD *)(v5 + 120);
        *(_DWORD *)(v19 + 128) = *(_DWORD *)(v5 + 128);
        *(_DWORD *)(v5 + 128) = 0;
      }
      if ( v5 + 136 != v79 )
      {
        v32 = *(_DWORD *)(v19 + 168) <= 0x40u;
        *(_QWORD *)(v19 + 136) = *(_QWORD *)(v5 + 136);
        *(_QWORD *)(v19 + 144) = *(_QWORD *)(v5 + 144);
        *(_DWORD *)(v19 + 152) = *(_DWORD *)(v5 + 152);
        if ( !v32 )
        {
          v34 = *(_QWORD *)(v19 + 160);
          if ( v34 )
          {
            v78 = v31;
            v80 = v18;
            j_j___libc_free_0_0(v34);
            v31 = v78;
            v18 = v80;
          }
        }
        *(_QWORD *)(v19 + 160) = *(_QWORD *)(v5 + 160);
        *(_DWORD *)(v19 + 168) = *(_DWORD *)(v5 + 168);
        *(_DWORD *)(v5 + 168) = 0;
      }
      *(_DWORD *)(v19 + 176) = *(_DWORD *)(v5 + 176);
      *(_QWORD *)(v19 + 184) = *(_QWORD *)(v5 + 184);
      if ( v10 >= v74 )
      {
        v4 = a1;
        if ( v64 )
          goto LABEL_45;
        goto LABEL_64;
      }
      v79 = v5 + 136;
      v7 = v18;
      v6 = v10;
      v76 = v31;
    }
    if ( (*(_BYTE *)(v5 + 16) & 1) == 0 )
    {
      v38 = *(_QWORD *)(v19 + 24);
      *(_QWORD *)(v19 + 24) = *(_QWORD *)(v5 + 24);
      v39 = *(_DWORD *)(v5 + 32);
      *(_QWORD *)(v5 + 24) = v38;
      LODWORD(v38) = *(_DWORD *)(v19 + 32);
      *(_DWORD *)(v19 + 32) = v39;
      *(_DWORD *)(v5 + 32) = v38;
      goto LABEL_25;
    }
    v26 = v7;
    v27 = (__int64 *)(v19 + 24);
    v22 = v5 + 24;
    v7 = v18;
LABEL_22:
    *(_BYTE *)(v26 + 8) |= 1u;
    v28 = *(_QWORD *)(v26 + 16);
    v29 = *(_DWORD *)(v26 + 24);
    for ( i = 0; i != 8; ++i )
      v27[i] = *(_QWORD *)(v22 + i * 8);
    *(_BYTE *)(v7 + 8) &= ~1u;
    *(_QWORD *)(v7 + 16) = v28;
    *(_DWORD *)(v7 + 24) = v29;
    goto LABEL_25;
  }
  if ( (a3 & 1) != 0 )
  {
    sub_28B4EA0((__int64)v81, a4);
    goto LABEL_53;
  }
  v10 = a2;
LABEL_64:
  if ( (a3 - 2) / 2 == v10 )
  {
    v10 = 2 * v10 + 1;
    v42 = v5;
    v5 = v4 + 192 * v10;
    sub_28B56D0(v42, v5);
  }
LABEL_45:
  sub_28B4EA0((__int64)v81, a4);
  v40 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    while ( 1 )
    {
      v5 = v4 + 192 * v40;
      if ( *(_DWORD *)(v5 + 112) == v84 )
      {
        if ( (int)sub_C4C880(v5 + 120, (__int64)&v85) < 0 )
          goto LABEL_67;
        v41 = (unsigned int)sub_C4C880((__int64)&v85, v5 + 120) >> 31;
      }
      else
      {
        if ( *(_DWORD *)(v5 + 112) < v84 )
          goto LABEL_67;
        LOBYTE(v41) = *(_DWORD *)(v5 + 112) > v84;
      }
      if ( (_BYTE)v41 )
        goto LABEL_52;
      if ( *(_DWORD *)(v5 + 152) == v87 )
      {
        if ( (int)sub_C4C880(v5 + 160, (__int64)&v88) >= 0 )
        {
LABEL_52:
          v5 = v4 + 192 * v10;
          break;
        }
      }
      else if ( *(_DWORD *)(v5 + 152) >= v87 )
      {
        goto LABEL_52;
      }
LABEL_67:
      v43 = v5 + 8;
      v44 = v4 + 192 * v10;
      *(_QWORD *)v44 = *(_QWORD *)v5;
      v45 = v44 + 8;
      if ( (*(_BYTE *)(v44 + 16) & 1) == 0 )
      {
        sub_C7D6A0(*(_QWORD *)(v44 + 24), 8LL * *(unsigned int *)(v44 + 32), 8);
        v43 = v5 + 8;
      }
      *(_QWORD *)(v44 + 16) = 1;
      v46 = (__int64 *)(v44 + 24);
      v47 = (__int64 *)(v44 + 88);
      v48 = v44 + 24;
      v49 = (__int64 *)(v44 + 24);
      do
      {
        if ( v49 )
          *v49 = -4096;
        ++v49;
      }
      while ( v49 != v47 );
      v50 = *(_DWORD *)(v5 + 16) & 0xFFFFFFFE;
      *(_DWORD *)(v5 + 16) = *(_DWORD *)(v44 + 16) & 0xFFFFFFFE | *(_DWORD *)(v5 + 16) & 1;
      *(_DWORD *)(v44 + 16) = *(_DWORD *)(v44 + 16) & 1 | v50;
      v51 = *(_DWORD *)(v44 + 20);
      *(_DWORD *)(v44 + 20) = *(_DWORD *)(v5 + 20);
      *(_DWORD *)(v5 + 20) = v51;
      if ( (*(_BYTE *)(v44 + 16) & 1) == 0 )
      {
        if ( (*(_BYTE *)(v5 + 16) & 1) == 0 )
        {
          v60 = *(_QWORD *)(v44 + 24);
          *(_QWORD *)(v44 + 24) = *(_QWORD *)(v5 + 24);
          v61 = *(_DWORD *)(v5 + 32);
          *(_QWORD *)(v5 + 24) = v60;
          LODWORD(v60) = *(_DWORD *)(v44 + 32);
          *(_DWORD *)(v44 + 32) = v61;
          *(_DWORD *)(v5 + 32) = v60;
          goto LABEL_79;
        }
        v52 = v43;
        v53 = (__int64 *)(v44 + 24);
        v48 = v5 + 24;
        v43 = v44 + 8;
        v45 = v52;
LABEL_76:
        *(_BYTE *)(v43 + 8) |= 1u;
        v54 = *(_QWORD *)(v43 + 16);
        v55 = 0;
        v56 = *(_DWORD *)(v43 + 24);
        do
        {
          v53[v55] = *(_QWORD *)(v48 + v55 * 8);
          ++v55;
        }
        while ( v55 != 8 );
        *(_BYTE *)(v45 + 8) &= ~1u;
        *(_QWORD *)(v45 + 16) = v54;
        *(_DWORD *)(v45 + 24) = v56;
        goto LABEL_79;
      }
      v53 = (__int64 *)(v5 + 24);
      if ( (*(_BYTE *)(v5 + 16) & 1) == 0 )
        goto LABEL_76;
      do
      {
        v59 = *v46;
        *v46++ = *v53;
        *v53++ = v59;
      }
      while ( v47 != v46 );
LABEL_79:
      *(_BYTE *)(v44 + 88) = *(_BYTE *)(v5 + 88);
      *(_DWORD *)(v44 + 92) = *(_DWORD *)(v5 + 92);
      if ( v5 + 96 != v44 + 96 )
      {
        v32 = *(_DWORD *)(v44 + 128) <= 0x40u;
        *(_QWORD *)(v44 + 96) = *(_QWORD *)(v5 + 96);
        *(_QWORD *)(v44 + 104) = *(_QWORD *)(v5 + 104);
        *(_DWORD *)(v44 + 112) = *(_DWORD *)(v5 + 112);
        if ( !v32 )
        {
          v57 = *(_QWORD *)(v44 + 120);
          if ( v57 )
            j_j___libc_free_0_0(v57);
        }
        *(_QWORD *)(v44 + 120) = *(_QWORD *)(v5 + 120);
        *(_DWORD *)(v44 + 128) = *(_DWORD *)(v5 + 128);
        *(_DWORD *)(v5 + 128) = 0;
      }
      if ( v5 + 136 != v44 + 136 )
      {
        v32 = *(_DWORD *)(v44 + 168) <= 0x40u;
        *(_QWORD *)(v44 + 136) = *(_QWORD *)(v5 + 136);
        *(_QWORD *)(v44 + 144) = *(_QWORD *)(v5 + 144);
        *(_DWORD *)(v44 + 152) = *(_DWORD *)(v5 + 152);
        if ( !v32 )
        {
          v58 = *(_QWORD *)(v44 + 160);
          if ( v58 )
            j_j___libc_free_0_0(v58);
        }
        *(_QWORD *)(v44 + 160) = *(_QWORD *)(v5 + 160);
        *(_DWORD *)(v44 + 168) = *(_DWORD *)(v5 + 168);
        *(_DWORD *)(v5 + 168) = 0;
      }
      v10 = v40;
      *(_DWORD *)(v44 + 176) = *(_DWORD *)(v5 + 176);
      *(_QWORD *)(v44 + 184) = *(_QWORD *)(v5 + 184);
      if ( a2 >= v40 )
        break;
      v40 = (v40 - 1) / 2;
    }
  }
LABEL_53:
  sub_28B56D0(v5, (__int64)v81);
  if ( v89 > 0x40 && v88 )
    j_j___libc_free_0_0(v88);
  if ( v86 > 0x40 && v85 )
    j_j___libc_free_0_0(v85);
  if ( (v81[16] & 1) == 0 )
    sub_C7D6A0(v82, 8LL * v83, 8);
}
