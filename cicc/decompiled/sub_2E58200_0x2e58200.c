// Function: sub_2E58200
// Address: 0x2e58200
//
void __fastcall sub_2E58200(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r14
  int v9; // ecx
  unsigned int v10; // esi
  int v11; // ecx
  unsigned int v12; // esi
  __int64 v13; // rax
  int v15; // r11d
  int v16; // r9d
  __int64 v17; // rbx
  int v18; // r9d
  __int64 v19; // r11
  int v20; // r14d
  int v21; // r10d
  unsigned int v22; // r12d
  int v23; // r13d
  int v24; // r8d
  __int64 v25; // rax
  char v26; // dl
  int v27; // eax
  __int64 v28; // r12
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // rdi
  int v32; // edx
  int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // rax
  unsigned int v37; // ecx
  int *v38; // rdx
  __int64 v39; // rax
  unsigned int v40; // eax
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rdx
  _QWORD *v44; // rcx
  int v46; // ebx
  int v47; // esi
  int *v48; // r12
  unsigned int v49; // edx
  int *v50; // r8
  int v51; // edi
  unsigned int v52; // ebx
  unsigned int v53; // r9d
  unsigned int v54; // esi
  int v55; // ebx
  unsigned __int64 v56; // rdi
  __int64 v57; // rdx
  unsigned __int64 v58; // rdi
  int *v61; // r10
  int v62; // eax
  int v63; // r11d
  int *v64; // r9
  int v65; // ecx
  int v66; // [rsp+10h] [rbp-110h]
  int v67; // [rsp+10h] [rbp-110h]
  int v68; // [rsp+10h] [rbp-110h]
  int v69; // [rsp+10h] [rbp-110h]
  int v70; // [rsp+10h] [rbp-110h]
  int v71; // [rsp+14h] [rbp-10Ch]
  int v72; // [rsp+14h] [rbp-10Ch]
  int v73; // [rsp+14h] [rbp-10Ch]
  int v74; // [rsp+14h] [rbp-10Ch]
  __int64 v75; // [rsp+18h] [rbp-108h]
  int v76; // [rsp+18h] [rbp-108h]
  __int64 v77; // [rsp+18h] [rbp-108h]
  __int64 v78; // [rsp+18h] [rbp-108h]
  __int64 v79; // [rsp+18h] [rbp-108h]
  __int64 v80; // [rsp+28h] [rbp-F8h]
  int v81; // [rsp+38h] [rbp-E8h]
  int v82; // [rsp+40h] [rbp-E0h]
  int v83; // [rsp+40h] [rbp-E0h]
  int v84; // [rsp+44h] [rbp-DCh]
  __int64 v85; // [rsp+48h] [rbp-D8h] BYREF
  unsigned int v86; // [rsp+54h] [rbp-CCh] BYREF
  int *v87; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v88; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v89; // [rsp+68h] [rbp-B8h]
  __int64 v90; // [rsp+70h] [rbp-B0h]
  unsigned int v91; // [rsp+78h] [rbp-A8h]
  __int64 v92; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v93; // [rsp+88h] [rbp-98h]
  __int64 v94; // [rsp+90h] [rbp-90h]
  __int64 v95; // [rsp+98h] [rbp-88h]
  _BYTE *v96; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v97; // [rsp+A8h] [rbp-78h]
  _BYTE v98[48]; // [rsp+B0h] [rbp-70h] BYREF
  int v99; // [rsp+E0h] [rbp-40h]

  v85 = a2;
  v3 = sub_2E57C80(a1 + 112, &v85);
  v8 = *v3;
  v82 = *(_DWORD *)(*v3 + 8);
  v84 = *(_DWORD *)(*v3 + 12);
  v96 = v98;
  v97 = 0x600000000LL;
  if ( *(_DWORD *)(v8 + 32) )
  {
    sub_2E4EFA0((__int64)&v96, v8 + 24, v4, v5, v6, v7);
    v40 = v97;
    v9 = *(_DWORD *)(v8 + 88);
    if ( *(_DWORD *)(v8 + 104) <= (unsigned int)v97 )
      v40 = *(_DWORD *)(v8 + 104);
    v99 = *(_DWORD *)(v8 + 88);
    if ( v40 )
    {
      v41 = 8LL * v40;
      v42 = 0;
      do
      {
        v43 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + v42);
        v44 = &v96[v42];
        v42 += 8;
        *v44 &= ~v43;
      }
      while ( v41 != v42 );
      v9 = v99;
    }
  }
  else
  {
    v9 = *(_DWORD *)(v8 + 88);
    v99 = v9;
  }
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  if ( v9 )
  {
    v10 = v9 - 1;
    v11 = -v9;
    v12 = v10 >> 6;
    v13 = 0;
    while ( 1 )
    {
      _RDX = *(_QWORD *)&v96[8 * v13];
      if ( v12 == (_DWORD)v13 )
        _RDX = (0xFFFFFFFFFFFFFFFFLL >> v11) & *(_QWORD *)&v96[8 * v13];
      if ( _RDX )
        break;
      if ( v12 + 1 == ++v13 )
        goto LABEL_9;
    }
    __asm { tzcnt   rdx, rdx }
    v46 = _RDX + ((_DWORD)v13 << 6);
    if ( v46 != -1 )
    {
      v47 = 0;
      v48 = (int *)(*(_QWORD *)(a1 + 88) + 4LL * v46);
LABEL_77:
      ++v92;
      v87 = 0;
LABEL_78:
      v47 *= 2;
LABEL_79:
      sub_A08C50((__int64)&v92, v47);
      sub_22B31A0((__int64)&v92, v48, &v87);
      v61 = v87;
      v62 = v94 + 1;
LABEL_86:
      LODWORD(v94) = v62;
      if ( *v61 != -1 )
        --HIDWORD(v94);
      *v61 = *v48;
LABEL_64:
      while ( 1 )
      {
        v52 = v46 + 1;
        if ( v99 == v52 )
          break;
        v53 = v52 >> 6;
        v54 = (unsigned int)(v99 - 1) >> 6;
        if ( v52 >> 6 > v54 )
          break;
        v55 = v52 & 0x3F;
        v56 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v55);
        if ( v55 == 0 )
          v56 = 0;
        v57 = v53;
        v58 = ~v56;
        while ( 1 )
        {
          _RAX = *(_QWORD *)&v96[8 * v57];
          if ( v53 == (_DWORD)v57 )
            _RAX = v58 & *(_QWORD *)&v96[8 * v57];
          if ( v54 == (_DWORD)v57 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v99;
          if ( _RAX )
            break;
          if ( v54 < (unsigned int)++v57 )
            goto LABEL_9;
        }
        __asm { tzcnt   rax, rax }
        v46 = _RAX + ((_DWORD)v57 << 6);
        if ( v46 == -1 )
          break;
        v47 = v95;
        v48 = (int *)(*(_QWORD *)(a1 + 88) + 4LL * v46);
        if ( !(_DWORD)v95 )
          goto LABEL_77;
        v49 = (v95 - 1) & (37 * *v48);
        v50 = (int *)(v93 + 4LL * v49);
        v51 = *v50;
        if ( *v48 != *v50 )
        {
          v63 = 1;
          v61 = 0;
          while ( v51 != -1 )
          {
            if ( v61 || v51 != -2 )
              v50 = v61;
            v49 = (v95 - 1) & (v63 + v49);
            v51 = *(_DWORD *)(v93 + 4LL * v49);
            if ( *v48 == v51 )
              goto LABEL_64;
            ++v63;
            v61 = v50;
            v50 = (int *)(v93 + 4LL * v49);
          }
          if ( !v61 )
            v61 = v50;
          ++v92;
          v62 = v94 + 1;
          v87 = v61;
          if ( 4 * ((int)v94 + 1) >= (unsigned int)(3 * v95) )
            goto LABEL_78;
          if ( (int)v95 - (v62 + HIDWORD(v94)) <= (unsigned int)v95 >> 3 )
            goto LABEL_79;
          goto LABEL_86;
        }
      }
    }
  }
LABEL_9:
  sub_2E52320(a1, v85, (__int64)&v88, (__int64)&v92);
  v15 = 0;
  v16 = 0;
  v17 = *(_QWORD *)(v85 + 56);
  if ( v85 + 48 == v17 )
    goto LABEL_37;
  v80 = v8;
  v18 = 0;
  v19 = v85 + 48;
  v20 = v82;
  v21 = 0;
  do
  {
    if ( *(_WORD *)(v17 + 68) == 68 || !*(_WORD *)(v17 + 68) )
      goto LABEL_34;
    v83 = 0;
    v81 = 0;
    v22 = *(unsigned __int8 *)(*(_QWORD *)(v17 + 16) + 4LL);
    v23 = *(_DWORD *)(v17 + 40) & 0xFFFFFF;
    if ( v22 == v23 )
      goto LABEL_21;
    v24 = v18;
    do
    {
      v25 = *(_QWORD *)(v17 + 32) + 40LL * v22;
      if ( *(_BYTE *)v25 )
        goto LABEL_19;
      v26 = *(_BYTE *)(v25 + 3);
      if ( (v26 & 0x20) != 0 )
        goto LABEL_19;
      v27 = *(_DWORD *)(v25 + 8);
      if ( v27 >= 0 || (v26 & 0x10) != 0 )
        goto LABEL_19;
      v86 = v27;
      if ( v91 )
      {
        v37 = (v91 - 1) & (37 * v27);
        v38 = (int *)(v89 + 16LL * v37);
        v76 = *v38;
        if ( *v38 == v27 )
        {
LABEL_53:
          if ( *((_QWORD *)v38 + 1) == v17 )
          {
            v67 = v21;
            v72 = v24;
            v77 = v19;
            v39 = sub_307B990(v86, *(_QWORD *)(a1 + 192), *(_QWORD *)(a1 + 176));
            v21 = v67;
            v81 += v39;
            v24 = v72;
            v19 = v77;
            v83 += HIDWORD(v39);
          }
          goto LABEL_19;
        }
        v68 = 1;
        v64 = 0;
        while ( v76 != -1 )
        {
          if ( v76 == -2 && !v64 )
            v64 = v38;
          v37 = (v91 - 1) & (v68 + v37);
          ++v68;
          v38 = (int *)(v89 + 16LL * v37);
          v76 = *v38;
          if ( v27 == *v38 )
            goto LABEL_53;
        }
        if ( !v64 )
          v64 = v38;
        ++v88;
        v65 = v90 + 1;
        v87 = v64;
        if ( 4 * ((int)v90 + 1) < 3 * v91 )
        {
          if ( v91 - HIDWORD(v90) - v65 <= v91 >> 3 )
          {
            v70 = v21;
            v74 = v24;
            v79 = v19;
            sub_2E51AA0((__int64)&v88, v91);
            sub_2E50670((__int64)&v88, (int *)&v86, &v87);
            v27 = v86;
            v64 = v87;
            v21 = v70;
            v24 = v74;
            v19 = v79;
            v65 = v90 + 1;
          }
          goto LABEL_100;
        }
      }
      else
      {
        ++v88;
        v87 = 0;
      }
      v69 = v21;
      v73 = v24;
      v78 = v19;
      sub_2E51AA0((__int64)&v88, 2 * v91);
      sub_2E50670((__int64)&v88, (int *)&v86, &v87);
      v27 = v86;
      v64 = v87;
      v19 = v78;
      v24 = v73;
      v21 = v69;
      v65 = v90 + 1;
LABEL_100:
      LODWORD(v90) = v65;
      if ( *v64 != -1 )
        --HIDWORD(v90);
      *v64 = v27;
      *((_QWORD *)v64 + 1) = 0;
LABEL_19:
      ++v22;
    }
    while ( v23 != v22 );
    v18 = v24;
    v23 = *(unsigned __int8 *)(*(_QWORD *)(v17 + 16) + 4LL);
LABEL_21:
    if ( v23 )
    {
      v28 = 0;
      v29 = 40LL * v23;
      do
      {
        v30 = v28 + *(_QWORD *)(v17 + 32);
        if ( !*(_BYTE *)v30 )
        {
          v31 = *(unsigned int *)(v30 + 8);
          if ( (int)v31 < 0 && *(_WORD *)(v17 + 68) != 10 && (*(_BYTE *)(v30 + 3) & 0x10) != 0 )
          {
            v66 = v21;
            v71 = v18;
            v75 = v19;
            v36 = sub_307B990(v31, *(_QWORD *)(a1 + 192), *(_QWORD *)(a1 + 176));
            v21 = v66;
            v18 = v71;
            v20 += v36;
            v19 = v75;
            v84 += HIDWORD(v36);
          }
        }
        v28 += 40;
      }
      while ( v29 != v28 );
    }
    if ( v21 < v20 )
      v21 = v20;
    if ( v18 < v84 )
      v18 = v84;
    v20 -= v81;
    v84 -= v83;
LABEL_34:
    if ( (*(_BYTE *)v17 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v17 + 44) & 8) != 0 )
        v17 = *(_QWORD *)(v17 + 8);
    }
    v17 = *(_QWORD *)(v17 + 8);
  }
  while ( v19 != v17 );
  v8 = v80;
  v15 = v18;
  v16 = v21;
LABEL_37:
  v32 = v16;
  v33 = v15;
  if ( *(_DWORD *)v8 >= v16 )
    v32 = *(_DWORD *)v8;
  if ( *(_DWORD *)(v8 + 4) >= v15 )
    v33 = *(_DWORD *)(v8 + 4);
  *(_DWORD *)v8 = v32;
  *(_DWORD *)(v8 + 4) = v33;
  if ( *(_DWORD *)(a1 + 28) >= v33 )
    v33 = *(_DWORD *)(a1 + 28);
  if ( *(_DWORD *)(a1 + 24) >= v32 )
    v32 = *(_DWORD *)(a1 + 24);
  v34 = (unsigned int)v95;
  *(_DWORD *)(a1 + 28) = v33;
  v35 = v93;
  *(_DWORD *)(a1 + 24) = v32;
  sub_C7D6A0(v35, 4 * v34, 4);
  sub_C7D6A0(v89, 16LL * v91, 8);
  if ( v96 != v98 )
    _libc_free((unsigned __int64)v96);
}
