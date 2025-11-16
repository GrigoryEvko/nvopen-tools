// Function: sub_CE4210
// Address: 0xce4210
//
__int64 __fastcall sub_CE4210(__int64 a1, __int64 a2)
{
  int *v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  int v8; // ecx
  unsigned int v9; // esi
  int v10; // ecx
  unsigned int v11; // esi
  __int64 v12; // rax
  __int64 v14; // r14
  _QWORD *v15; // rdx
  _QWORD *v16; // r11
  __int64 v17; // rax
  _QWORD *v18; // rbx
  _QWORD *v19; // r10
  _QWORD *v20; // r15
  int v21; // r10d
  unsigned int v22; // edi
  _QWORD *v23; // rax
  _QWORD *v24; // rcx
  _QWORD *v25; // r12
  int v26; // eax
  __int64 v27; // rsi
  int v28; // ecx
  int v29; // edi
  unsigned int v30; // eax
  unsigned int v31; // ecx
  int v32; // eax
  __int64 v33; // rdi
  int v34; // ecx
  __int64 v35; // rsi
  int v36; // ecx
  int v37; // edi
  unsigned int v38; // eax
  __int64 v39; // rax
  int v40; // ebx
  int v41; // esi
  int v42; // eax
  int v43; // edx
  int v44; // eax
  __int64 v45; // rsi
  __int64 v46; // rdi
  __int64 v47; // rsi
  __int64 result; // rax
  _QWORD *v49; // r8
  unsigned int v50; // r14d
  int v51; // r10d
  __int64 v52; // rsi
  __int64 v53; // rax
  int *v54; // rbx
  unsigned int v55; // eax
  __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rdx
  _QWORD *v59; // rcx
  int v61; // ebx
  int v62; // esi
  _QWORD *v63; // r12
  unsigned int v64; // eax
  _QWORD *v65; // r8
  __int64 v66; // rdi
  unsigned int v67; // eax
  unsigned int v68; // r10d
  unsigned int v69; // esi
  int v70; // ecx
  unsigned __int64 v71; // r8
  __int64 v72; // rdx
  unsigned __int64 v73; // r8
  __int64 v76; // rdx
  _QWORD *v77; // r10
  __int64 v78; // rcx
  int v79; // eax
  int v80; // r11d
  _QWORD *v81; // r9
  int v82; // r11d
  int v83; // r11d
  __int64 v84; // rdx
  __int64 v85; // rcx
  int v86; // r14d
  _QWORD *v87; // r9
  int *v88; // [rsp+8h] [rbp-108h]
  __int64 v89; // [rsp+10h] [rbp-100h]
  _QWORD *v90; // [rsp+20h] [rbp-F0h]
  _QWORD *v91; // [rsp+20h] [rbp-F0h]
  _QWORD *v92; // [rsp+20h] [rbp-F0h]
  __int64 v93; // [rsp+28h] [rbp-E8h]
  int v94; // [rsp+30h] [rbp-E0h]
  int v95; // [rsp+34h] [rbp-DCh]
  int v96; // [rsp+38h] [rbp-D8h]
  int v97; // [rsp+3Ch] [rbp-D4h]
  int v98; // [rsp+40h] [rbp-D0h]
  int v99; // [rsp+44h] [rbp-CCh]
  __int64 v100; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v101; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v102; // [rsp+58h] [rbp-B8h]
  __int64 v103; // [rsp+60h] [rbp-B0h]
  unsigned int v104; // [rsp+68h] [rbp-A8h]
  __int64 v105; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v106; // [rsp+78h] [rbp-98h]
  __int64 v107; // [rsp+80h] [rbp-90h]
  __int64 v108; // [rsp+88h] [rbp-88h]
  _BYTE *v109; // [rsp+90h] [rbp-80h] BYREF
  __int64 v110; // [rsp+98h] [rbp-78h]
  _BYTE v111[48]; // [rsp+A0h] [rbp-70h] BYREF
  int v112; // [rsp+D0h] [rbp-40h]

  v100 = a2;
  v3 = (int *)*sub_CE3FC0(a1 + 112, &v100);
  v7 = (unsigned int)v3[8];
  v88 = v3;
  v97 = v3[2];
  v96 = v3[3];
  v109 = v111;
  v110 = 0x600000000LL;
  if ( (_DWORD)v7 )
  {
    v54 = v3;
    sub_CE14D0((__int64)&v109, (__int64)(v3 + 6), v7, v4, v5, v6);
    v55 = v110;
    v8 = v54[22];
    if ( v54[26] <= (unsigned int)v110 )
      v55 = v54[26];
    v112 = v54[22];
    if ( v55 )
    {
      v56 = 8LL * v55;
      v57 = 0;
      do
      {
        v58 = *(_QWORD *)(*((_QWORD *)v54 + 12) + v57);
        v59 = &v109[v57];
        v57 += 8;
        *v59 &= ~v58;
      }
      while ( v56 != v57 );
      v8 = v112;
    }
  }
  else
  {
    v8 = v3[22];
    v112 = v8;
  }
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  if ( v8 )
  {
    v9 = v8 - 1;
    v10 = -v8;
    v11 = v9 >> 6;
    v12 = 0;
    while ( 1 )
    {
      _RDX = *(_QWORD *)&v109[8 * v12];
      if ( v11 == (_DWORD)v12 )
        _RDX = (0xFFFFFFFFFFFFFFFFLL >> v10) & *(_QWORD *)&v109[8 * v12];
      if ( _RDX )
        break;
      if ( v11 + 1 == ++v12 )
        goto LABEL_9;
    }
    __asm { tzcnt   rdx, rdx }
    v61 = _RDX + ((_DWORD)v12 << 6);
    if ( v61 != -1 )
    {
      v62 = 0;
      v63 = (_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL * v61);
LABEL_97:
      ++v105;
LABEL_98:
      sub_CE2A30((__int64)&v105, 2 * v62);
      if ( !(_DWORD)v108 )
        goto LABEL_151;
      LODWORD(v76) = (v108 - 1) & (((unsigned int)*v63 >> 9) ^ ((unsigned int)*v63 >> 4));
      v77 = (_QWORD *)(v106 + 8LL * (unsigned int)v76);
      v78 = *v77;
      v79 = v107 + 1;
      if ( *v77 == *v63 )
      {
        while ( 1 )
        {
LABEL_113:
          LODWORD(v107) = v79;
          if ( *v77 != -4096 )
            --HIDWORD(v107);
          *v77 = *v63;
          do
          {
LABEL_84:
            v67 = v61 + 1;
            if ( v112 == v61 + 1 )
              goto LABEL_9;
            v68 = v67 >> 6;
            v69 = (unsigned int)(v112 - 1) >> 6;
            if ( v67 >> 6 > v69 )
              goto LABEL_9;
            v70 = 64 - (v67 & 0x3F);
            v71 = 0xFFFFFFFFFFFFFFFFLL >> v70;
            v72 = v68;
            if ( v70 == 64 )
              v71 = 0;
            v73 = ~v71;
            while ( 1 )
            {
              _RAX = *(_QWORD *)&v109[8 * v72];
              if ( v68 == (_DWORD)v72 )
                _RAX = v73 & *(_QWORD *)&v109[8 * v72];
              if ( v69 == (_DWORD)v72 )
                _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v112;
              if ( _RAX )
                break;
              if ( v69 < (unsigned int)++v72 )
                goto LABEL_9;
            }
            __asm { tzcnt   rax, rax }
            v61 = ((_DWORD)v72 << 6) + _RAX;
            if ( v61 == -1 )
              goto LABEL_9;
            v62 = v108;
            v63 = (_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL * v61);
            if ( !(_DWORD)v108 )
              goto LABEL_97;
            v64 = (v108 - 1) & (((unsigned int)*v63 >> 9) ^ ((unsigned int)*v63 >> 4));
            v65 = (_QWORD *)(v106 + 8LL * v64);
            v66 = *v65;
          }
          while ( *v63 == *v65 );
          v82 = 1;
          v77 = 0;
          while ( v66 != -4096 )
          {
            if ( v77 || v66 != -8192 )
              v65 = v77;
            v64 = (v108 - 1) & (v82 + v64);
            v66 = *(_QWORD *)(v106 + 8LL * v64);
            if ( *v63 == v66 )
              goto LABEL_84;
            ++v82;
            v77 = v65;
            v65 = (_QWORD *)(v106 + 8LL * v64);
          }
          if ( !v77 )
            v77 = v65;
          ++v105;
          v79 = v107 + 1;
          if ( 4 * ((int)v107 + 1) >= (unsigned int)(3 * v108) )
            goto LABEL_98;
          if ( (int)v108 - (v79 + HIDWORD(v107)) <= (unsigned int)v108 >> 3 )
          {
            sub_CE2A30((__int64)&v105, v108);
            if ( !(_DWORD)v108 )
            {
LABEL_151:
              LODWORD(v107) = v107 + 1;
              BUG();
            }
            v83 = 1;
            v81 = 0;
            LODWORD(v84) = (v108 - 1) & (((unsigned int)*v63 >> 9) ^ ((unsigned int)*v63 >> 4));
            v77 = (_QWORD *)(v106 + 8LL * (unsigned int)v84);
            v85 = *v77;
            v79 = v107 + 1;
            if ( *v77 != *v63 )
              break;
          }
        }
        while ( v85 != -4096 )
        {
          if ( v85 == -8192 && !v81 )
            v81 = v77;
          v84 = ((_DWORD)v108 - 1) & (unsigned int)(v84 + v83);
          v77 = (_QWORD *)(v106 + 8 * v84);
          v85 = *v77;
          if ( *v63 == *v77 )
            goto LABEL_113;
          ++v83;
        }
      }
      else
      {
        v80 = 1;
        v81 = 0;
        while ( v78 != -4096 )
        {
          if ( !v81 && v78 == -8192 )
            v81 = v77;
          v76 = ((_DWORD)v108 - 1) & (unsigned int)(v76 + v80);
          v77 = (_QWORD *)(v106 + 8 * v76);
          v78 = *v77;
          if ( *v63 == *v77 )
            goto LABEL_113;
          ++v80;
        }
      }
      if ( v81 )
        v77 = v81;
      goto LABEL_113;
    }
  }
LABEL_9:
  sub_CE2C00(a1, v100, (__int64)&v101, (__int64)&v105);
  v98 = 0;
  v99 = 0;
  v14 = *(_QWORD *)(v100 + 56);
  v93 = v100 + 48;
  if ( v100 + 48 != v14 )
  {
    while ( 1 )
    {
      if ( !v14 )
        BUG();
      if ( *(_BYTE *)(v14 - 24) == 84 )
        goto LABEL_41;
      if ( sub_CE16D0(v14 - 24) )
      {
        v17 = 32LL * (*(_DWORD *)(v14 - 20) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v14 - 17) & 0x40) != 0 )
        {
          v18 = *(_QWORD **)(v14 - 32);
          v19 = &v18[(unsigned __int64)v17 / 8];
        }
        else
        {
          v19 = v16;
          v18 = &v16[v17 / 0xFFFFFFFFFFFFFFF8LL];
        }
        v95 = 0;
        v94 = 0;
        if ( v18 != v19 )
        {
          v89 = v14;
          v20 = v19;
          while ( 1 )
          {
            v25 = (_QWORD *)*v18;
            if ( ((*(_BYTE *)(*(_QWORD *)(*v18 + 8LL) + 8LL) - 15) & 0xFD) != 0 )
              goto LABEL_24;
            v26 = *(_DWORD *)(a1 + 80);
            v27 = *(_QWORD *)(a1 + 64);
            if ( v26 )
            {
              v28 = v26 - 1;
              v29 = 1;
              v30 = (v26 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              v15 = *(_QWORD **)(v27 + 16LL * v30);
              if ( v15 != v25 )
              {
                while ( v15 != (_QWORD *)-4096LL )
                {
                  v30 = v28 & (v29 + v30);
                  v15 = *(_QWORD **)(v27 + 16LL * v30);
                  if ( v25 == v15 )
                    goto LABEL_24;
                  ++v29;
                }
                goto LABEL_20;
              }
LABEL_24:
              if ( !v104 )
              {
                ++v101;
                goto LABEL_26;
              }
              v21 = 1;
              v15 = 0;
              v22 = (v104 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              v23 = (_QWORD *)(v102 + 16LL * v22);
              v24 = (_QWORD *)*v23;
              if ( v25 == (_QWORD *)*v23 )
              {
LABEL_18:
                if ( v16 == (_QWORD *)v23[1] )
                {
                  v92 = v16;
                  v53 = sub_FCD870(v25, *(_QWORD *)(*(_QWORD *)a1 + 40LL) + 312LL, v15);
                  v16 = v92;
                  v94 += v53;
                  v95 += HIDWORD(v53);
                }
                goto LABEL_20;
              }
              while ( v24 != (_QWORD *)-4096LL )
              {
                if ( v24 == (_QWORD *)-8192LL && !v15 )
                  v15 = v23;
                v22 = (v104 - 1) & (v21 + v22);
                v23 = (_QWORD *)(v102 + 16LL * v22);
                v24 = (_QWORD *)*v23;
                if ( v25 == (_QWORD *)*v23 )
                  goto LABEL_18;
                ++v21;
              }
              if ( !v15 )
                v15 = v23;
              ++v101;
              v32 = v103 + 1;
              if ( 4 * ((int)v103 + 1) < 3 * v104 )
              {
                if ( v104 - HIDWORD(v103) - v32 <= v104 >> 3 )
                {
                  v91 = v16;
                  sub_CE25F0((__int64)&v101, v104);
                  if ( !v104 )
                  {
LABEL_153:
                    LODWORD(v103) = v103 + 1;
                    BUG();
                  }
                  v49 = 0;
                  v50 = (v104 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
                  v16 = v91;
                  v51 = 1;
                  v32 = v103 + 1;
                  v15 = (_QWORD *)(v102 + 16LL * v50);
                  v52 = *v15;
                  if ( v25 != (_QWORD *)*v15 )
                  {
                    while ( v52 != -4096 )
                    {
                      if ( !v49 && v52 == -8192 )
                        v49 = v15;
                      v50 = (v104 - 1) & (v51 + v50);
                      v15 = (_QWORD *)(v102 + 16LL * v50);
                      v52 = *v15;
                      if ( v25 == (_QWORD *)*v15 )
                        goto LABEL_28;
                      ++v51;
                    }
                    if ( v49 )
                      v15 = v49;
                  }
                }
                goto LABEL_28;
              }
LABEL_26:
              v90 = v16;
              sub_CE25F0((__int64)&v101, 2 * v104);
              if ( !v104 )
                goto LABEL_153;
              v16 = v90;
              v31 = (v104 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              v32 = v103 + 1;
              v15 = (_QWORD *)(v102 + 16LL * v31);
              v33 = *v15;
              if ( (_QWORD *)*v15 != v25 )
              {
                v86 = 1;
                v87 = 0;
                while ( v33 != -4096 )
                {
                  if ( !v87 && v33 == -8192 )
                    v87 = v15;
                  v31 = (v104 - 1) & (v86 + v31);
                  v15 = (_QWORD *)(v102 + 16LL * v31);
                  v33 = *v15;
                  if ( v25 == (_QWORD *)*v15 )
                    goto LABEL_28;
                  ++v86;
                }
                if ( v87 )
                  v15 = v87;
              }
LABEL_28:
              LODWORD(v103) = v32;
              if ( *v15 != -4096 )
                --HIDWORD(v103);
              v18 += 4;
              *v15 = v25;
              v15[1] = 0;
              if ( v20 == v18 )
              {
LABEL_31:
                v14 = v89;
                break;
              }
            }
            else
            {
LABEL_20:
              v18 += 4;
              if ( v20 == v18 )
                goto LABEL_31;
            }
          }
        }
      }
      else
      {
        v95 = 0;
        v94 = 0;
      }
      if ( ((*(_BYTE *)(*(_QWORD *)(v14 - 16) + 8LL) - 15) & 0xFD) != 0 )
        goto LABEL_35;
      v34 = *(_DWORD *)(a1 + 80);
      v35 = *(_QWORD *)(a1 + 64);
      if ( v34 )
        break;
LABEL_36:
      v40 = v99;
      v41 = v98;
      if ( v99 < v97 )
        v40 = v97;
      v99 = v40;
      if ( v98 < v96 )
        v41 = v96;
      v98 = v41;
      v97 -= v94;
      v96 -= v95;
LABEL_41:
      v14 = *(_QWORD *)(v14 + 8);
      if ( v93 == v14 )
        goto LABEL_42;
    }
    v36 = v34 - 1;
    v37 = 1;
    v38 = v36 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v15 = *(_QWORD **)(v35 + 16LL * v38);
    if ( v16 != v15 )
    {
      while ( v15 != (_QWORD *)-4096LL )
      {
        v38 = v36 & (v37 + v38);
        v15 = *(_QWORD **)(v35 + 16LL * v38);
        if ( v16 == v15 )
          goto LABEL_35;
        ++v37;
      }
      goto LABEL_36;
    }
LABEL_35:
    v39 = sub_FCD870(v16, *(_QWORD *)(*(_QWORD *)a1 + 40LL) + 312LL, v15);
    v97 += v39;
    v96 += HIDWORD(v39);
    goto LABEL_36;
  }
LABEL_42:
  v42 = v99;
  if ( *v88 >= v99 )
    v42 = *v88;
  v43 = v42;
  v44 = v98;
  if ( v88[1] >= v98 )
    v44 = v88[1];
  *v88 = v43;
  v88[1] = v44;
  if ( *(_DWORD *)(a1 + 28) >= v44 )
    v44 = *(_DWORD *)(a1 + 28);
  if ( *(_DWORD *)(a1 + 24) >= v43 )
    v43 = *(_DWORD *)(a1 + 24);
  v45 = (unsigned int)v108;
  *(_DWORD *)(a1 + 28) = v44;
  v46 = v106;
  *(_DWORD *)(a1 + 24) = v43;
  sub_C7D6A0(v46, 8 * v45, 8);
  v47 = 16LL * v104;
  result = sub_C7D6A0(v102, v47, 8);
  if ( v109 != v111 )
    return _libc_free(v109, v47);
  return result;
}
