// Function: sub_2A93680
// Address: 0x2a93680
//
__int64 __fastcall sub_2A93680(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, int a5)
{
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 *v11; // rcx
  __int64 v12; // r10
  __int16 v13; // ax
  __int64 v14; // rsi
  unsigned int v15; // ebx
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned int v22; // esi
  unsigned __int8 *v23; // r12
  unsigned __int8 v24; // al
  int v25; // eax
  unsigned int v26; // esi
  __int64 v29; // rdx
  const void **v30; // rsi
  int v31; // ecx
  int v32; // r11d
  unsigned int v33; // r14d
  bool v34; // r12
  __int64 v35; // rbx
  unsigned int v36; // esi
  __int64 v37; // r8
  _BYTE *v38; // r10
  int v39; // r11d
  __int64 v40; // rdx
  _BYTE *v41; // rax
  __int64 v42; // rcx
  _QWORD *v43; // r15
  __int64 v44; // r8
  unsigned int v45; // ebx
  unsigned int v46; // eax
  _QWORD *v47; // rbx
  _BYTE *v48; // r8
  size_t v49; // r15
  _QWORD *v50; // r15
  unsigned int v51; // ebx
  unsigned int v52; // r12d
  unsigned int v53; // ecx
  unsigned __int64 v54; // rax
  __int64 v55; // rax
  _QWORD *v56; // rdi
  unsigned __int64 v57; // rax
  int v58; // edx
  int v59; // eax
  unsigned __int64 v60; // rax
  _QWORD *v61; // rax
  unsigned int v62; // eax
  int v63; // eax
  int v64; // eax
  __int64 v65; // rdx
  __int64 *v66; // rax
  __int64 v67; // rdx
  int v68; // r8d
  __int64 v69; // rsi
  __int64 v70; // rbx
  unsigned int v71; // r15d
  __int64 v72; // rax
  char v73; // dl
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rax
  unsigned __int64 v76; // rdx
  int v77; // edx
  bool v78; // zf
  int v79; // eax
  unsigned __int8 v80; // al
  unsigned int v81; // ebx
  __int64 v84; // rdx
  unsigned __int64 v85; // rax
  unsigned __int64 v86; // rdx
  int v87; // edx
  int v88; // eax
  int v89; // eax
  unsigned int v90; // r14d
  _QWORD *v91; // [rsp-D0h] [rbp-D0h]
  unsigned int v92; // [rsp-C8h] [rbp-C8h]
  _QWORD *v93; // [rsp-C8h] [rbp-C8h]
  unsigned __int8 v95; // [rsp-C0h] [rbp-C0h]
  __int64 v96; // [rsp-C0h] [rbp-C0h]
  __int64 v97; // [rsp-C0h] [rbp-C0h]
  unsigned __int8 v98; // [rsp-C0h] [rbp-C0h]
  _BYTE *v99; // [rsp-C0h] [rbp-C0h]
  __int64 v100; // [rsp-C0h] [rbp-C0h]
  unsigned __int8 v101; // [rsp-C0h] [rbp-C0h]
  unsigned __int8 v102; // [rsp-C0h] [rbp-C0h]
  __int64 v103; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v104; // [rsp-B0h] [rbp-B0h] BYREF
  __m128i v105; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i v106; // [rsp-98h] [rbp-98h] BYREF
  unsigned __int64 *v107; // [rsp-88h] [rbp-88h] BYREF
  unsigned int v108; // [rsp-80h] [rbp-80h]
  unsigned __int64 v109; // [rsp-78h] [rbp-78h] BYREF
  unsigned __int64 *v110; // [rsp-70h] [rbp-70h]
  _QWORD v111[2]; // [rsp-68h] [rbp-68h] BYREF
  __int64 v112; // [rsp-58h] [rbp-58h]
  __int64 v113; // [rsp-50h] [rbp-50h]
  __int64 v114; // [rsp-48h] [rbp-48h]

  if ( !a2 || a5 == 10 )
    return 0;
  v8 = a2;
  v9 = *(unsigned int *)(a3 + 24);
  v10 = *(_QWORD *)(a3 + 8);
  if ( (_DWORD)v9 )
  {
    a2 = ((_DWORD)v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v11 = (__int64 *)(v10 + 16LL * (unsigned int)a2);
    v12 = *v11;
    if ( v8 == *v11 )
    {
LABEL_6:
      if ( v11 != (__int64 *)(v10 + 16 * v9) )
        return *((unsigned __int8 *)v11 + 8);
    }
    else
    {
      v31 = 1;
      while ( v12 != -4096 )
      {
        v32 = v31 + 1;
        a2 = ((_DWORD)v9 - 1) & (unsigned int)(v31 + a2);
        v11 = (__int64 *)(v10 + 16LL * (unsigned int)a2);
        v12 = *v11;
        if ( v8 == *v11 )
          goto LABEL_6;
        v31 = v32;
      }
    }
  }
  v105.m128i_i64[0] = a3;
  v13 = *(_WORD *)(v8 + 24);
  v105.m128i_i64[1] = v8;
  v103 = a3;
  v104 = v8;
  v106 = _mm_loadu_si128(&v105);
  if ( !v13 )
  {
    v14 = *(_QWORD *)(v8 + 32);
    v15 = *(_DWORD *)(v14 + 32);
    v16 = *(_QWORD *)(v14 + 24);
    v17 = 1LL << ((unsigned __int8)v15 - 1);
    if ( v15 > 0x40 )
    {
      v30 = (const void **)(v14 + 24);
      if ( (*(_QWORD *)(v16 + 8LL * ((v15 - 1) >> 6)) & v17) == 0 )
      {
        v108 = v15;
        sub_C43780((__int64)&v107, v30);
        v15 = v108;
LABEL_16:
        LODWORD(v110) = v15;
        if ( v15 > 0x40 )
        {
          sub_C43780((__int64)&v109, (const void **)&v107);
          v15 = (unsigned int)v110;
          if ( (unsigned int)v110 > 0x40 )
          {
            if ( (unsigned int)sub_C44630((__int64)&v109) == 1 )
            {
LABEL_19:
              v21 = *(_QWORD *)v109;
LABEL_20:
              v22 = -1;
              if ( v21 )
              {
                _BitScanReverse64(&v21, v21);
                v22 = 63 - (v21 ^ 0x3F);
              }
              v95 = sub_2A931D0(v106.m128i_i64, v22);
              sub_969240((__int64 *)&v109);
              sub_969240((__int64 *)&v107);
              return v95;
            }
            v33 = sub_C44590((__int64)&v109);
            memset((void *)v109, 0, 8 * (((unsigned __int64)v15 + 63) >> 6));
            v29 = 1LL << v33;
            if ( (unsigned int)v110 > 0x40 )
            {
              *(_QWORD *)(v109 + 8LL * (v33 >> 6)) |= v29;
              if ( (unsigned int)v110 > 0x40 )
                goto LABEL_19;
LABEL_38:
              v21 = v109;
              goto LABEL_20;
            }
            goto LABEL_37;
          }
LABEL_31:
          _RAX = v109;
          if ( v109 )
          {
            if ( (v109 & (v109 - 1)) == 0 )
              goto LABEL_38;
            __asm { tzcnt   rax, rax }
          }
          else
          {
            LODWORD(_RAX) = 64;
          }
          v109 = 0;
          if ( (unsigned int)_RAX <= v15 )
            LOBYTE(v15) = _RAX;
          v29 = 1LL << v15;
LABEL_37:
          v109 |= v29;
          goto LABEL_38;
        }
LABEL_30:
        v109 = (unsigned __int64)v107;
        goto LABEL_31;
      }
      LODWORD(v110) = v15;
      sub_C43780((__int64)&v109, v30);
      v15 = (unsigned int)v110;
      if ( (unsigned int)v110 > 0x40 )
      {
        sub_C43D10((__int64)&v109);
LABEL_15:
        sub_C46250((__int64)&v109);
        v15 = (unsigned int)v110;
        v108 = (unsigned int)v110;
        v107 = (unsigned __int64 *)v109;
        goto LABEL_16;
      }
      v18 = v109;
    }
    else
    {
      v18 = *(_QWORD *)(v14 + 24);
      if ( (v17 & v16) == 0 )
      {
        v108 = *(_DWORD *)(v14 + 32);
        v107 = (unsigned __int64 *)v16;
        LODWORD(v110) = v15;
        goto LABEL_30;
      }
      LODWORD(v110) = *(_DWORD *)(v14 + 32);
    }
    v19 = ~v18 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v15);
    v20 = 0;
    if ( v15 )
      v20 = v19;
    v109 = v20;
    goto LABEL_15;
  }
  if ( v13 != 15 )
  {
    v34 = v13 == 14 || (unsigned __int16)(v13 - 2) <= 2u;
    if ( v34 )
    {
      v26 = sub_2A93680(a1, *(_QWORD *)(v8 + 32), a3, a4, (unsigned int)(a5 + 1));
      return sub_2A931D0(v105.m128i_i64, v26);
    }
    if ( v13 == 5 )
    {
      v43 = *(_QWORD **)(v8 + 32);
      v91 = &v43[*(_QWORD *)(v8 + 40)];
      if ( v43 == v91 )
      {
        v26 = 0;
      }
      else
      {
        v44 = (unsigned int)(a5 + 1);
        v45 = 0;
        while ( 1 )
        {
          v92 = v44;
          v96 = a3;
          v46 = sub_2A93680(a1, *v43, a3, a4, v44);
          v26 = v46;
          if ( !(_BYTE)v46 )
            break;
          a3 = v96;
          v44 = v92;
          if ( v34 )
          {
            if ( (unsigned __int8)v45 > (unsigned __int8)v46 )
              v45 = v46;
          }
          else
          {
            v45 = v46;
          }
          ++v43;
          v34 = 1;
          if ( v91 == v43 )
          {
            v26 = v45;
            return sub_2A931D0(v105.m128i_i64, v26);
          }
        }
      }
      return sub_2A931D0(v105.m128i_i64, v26);
    }
    if ( v13 == 6 )
    {
      v50 = *(_QWORD **)(v8 + 32);
      v93 = &v50[*(_QWORD *)(v8 + 40)];
      if ( v50 == v93 )
      {
        v52 = 0;
      }
      else
      {
        v51 = a5 + 1;
        v52 = 0;
        do
        {
          v97 = a3;
          v53 = (unsigned __int8)v52 + (unsigned __int8)sub_2A93680(a1, *v50, a3, a4, v51);
          if ( v53 > 0x3F )
            break;
          ++v50;
          a3 = v97;
          _BitScanReverse64(&v54, 1LL << v53);
          v52 = 63 - (v54 ^ 0x3F);
        }
        while ( v93 != v50 );
      }
      return sub_2A931D0(v105.m128i_i64, v52);
    }
    if ( v13 != 8 )
      return sub_2A93420(&v103);
    if ( *(_QWORD *)(v8 + 40) != 2 )
      return sub_2A93420(&v103);
    v55 = *(_QWORD *)(*(_QWORD *)(v8 + 32) + 8LL);
    if ( *(_WORD *)(v55 + 24) )
      return sub_2A93420(&v103);
    v100 = a3;
    sub_9692E0((__int64)&v107, (__int64 *)(*(_QWORD *)(v55 + 32) + 24LL));
    v66 = *(__int64 **)(v8 + 32);
    v67 = v100;
    v68 = a5;
    v69 = *v66;
    if ( *(_WORD *)(*v66 + 24) )
      goto LABEL_122;
    v70 = *(_QWORD *)(v69 + 32);
    v71 = *(_DWORD *)(v70 + 32);
    if ( v71 > 0x40 )
    {
      v89 = sub_C444A0(v70 + 24);
      v67 = v100;
      v68 = a5;
      if ( v71 - v89 > 0x40 )
        goto LABEL_122;
      v72 = **(_QWORD **)(v70 + 24);
    }
    else
    {
      v72 = *(_QWORD *)(v70 + 24);
    }
    if ( v72 )
    {
LABEL_122:
      v73 = sub_2A93680(a1, v69, v67, a4, (unsigned int)(v68 + 1));
      v74 = (unsigned __int64)v107;
      if ( v108 > 0x40 )
        v74 = *v107;
      v75 = -(__int64)(v74 | (1LL << v73)) & (v74 | (1LL << v73));
      _BitScanReverse64(&v76, v75);
      v77 = v76 ^ 0x3F;
      v78 = v75 == 0;
      v79 = 64;
      if ( !v78 )
        v79 = v77;
      v80 = sub_2A931D0(v105.m128i_i64, 63 - v79);
LABEL_127:
      v101 = v80;
      sub_969240((__int64 *)&v107);
      return v101;
    }
    sub_9865C0((__int64)&v109, (__int64)&v107);
    v81 = (unsigned int)v110;
    if ( (unsigned int)v110 <= 0x40 )
    {
      _RAX = v109;
      if ( v109 )
      {
        if ( (v109 & (v109 - 1)) == 0 )
        {
LABEL_138:
          v85 = v109;
LABEL_139:
          _BitScanReverse64(&v86, v85);
          v87 = v86 ^ 0x3F;
          v78 = v85 == 0;
          v88 = 64;
          if ( !v78 )
            v88 = v87;
          v102 = sub_2A931D0(v106.m128i_i64, 63 - v88);
          sub_969240((__int64 *)&v109);
          v80 = v102;
          goto LABEL_127;
        }
        __asm { tzcnt   rax, rax }
      }
      else
      {
        LODWORD(_RAX) = 64;
      }
      v109 = 0;
      if ( (unsigned int)v110 <= (unsigned int)_RAX )
        LOBYTE(_RAX) = (_BYTE)v110;
      v84 = 1LL << _RAX;
LABEL_137:
      v109 |= v84;
      goto LABEL_138;
    }
    if ( (unsigned int)sub_C44630((__int64)&v109) != 1 )
    {
      v90 = sub_C44590((__int64)&v109);
      memset((void *)v109, 0, 8 * (((unsigned __int64)v81 + 63) >> 6));
      v84 = 1LL << v90;
      if ( (unsigned int)v110 <= 0x40 )
        goto LABEL_137;
      *(_QWORD *)(v109 + 8LL * (v90 >> 6)) |= v84;
      if ( (unsigned int)v110 <= 0x40 )
        goto LABEL_138;
    }
    v85 = *(_QWORD *)v109;
    goto LABEL_139;
  }
  v23 = sub_BD3990(*(unsigned __int8 **)(v8 - 8), a2);
  v24 = *v23;
  if ( *v23 != 20 )
    goto LABEL_25;
  v47 = *(_QWORD **)(*a1 + 40);
  v109 = (unsigned __int64)v111;
  v48 = (_BYTE *)v47[29];
  v49 = v47[30];
  if ( &v48[v49] && !v48 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v107 = (unsigned __int64 *)v47[30];
  if ( v49 > 0xF )
  {
    v99 = v48;
    v61 = (_QWORD *)sub_22409D0((__int64)&v109, (unsigned __int64 *)&v107, 0);
    v48 = v99;
    v109 = (unsigned __int64)v61;
    v56 = v61;
    v111[0] = v107;
    goto LABEL_91;
  }
  if ( v49 != 1 )
  {
    if ( !v49 )
      goto LABEL_75;
    v56 = v111;
LABEL_91:
    memcpy(v56, v48, v49);
    goto LABEL_75;
  }
  LOBYTE(v111[0]) = *v48;
LABEL_75:
  v110 = v107;
  *((_BYTE *)v107 + v109) = 0;
  v112 = v47[33];
  v113 = v47[34];
  v114 = v47[35];
  if ( (_DWORD)v112 != 21 )
  {
    if ( (_QWORD *)v109 != v111 )
      j_j___libc_free_0(v109);
    v24 = *v23;
LABEL_25:
    if ( v24 == 3 )
    {
      v25 = (*((_WORD *)v23 + 17) >> 1) & 0x3F;
      v26 = v25 - 1;
      if ( !(_WORD)v25 )
        v26 = 0;
      return sub_2A931D0(v105.m128i_i64, v26);
    }
    if ( v24 == 60 )
    {
      _BitScanReverse64(&v60, 1LL << *((_WORD *)v23 + 1));
      return sub_2A931D0(v105.m128i_i64, 63 - ((unsigned int)v60 ^ 0x3F));
    }
    if ( v24 == 22 && *(_BYTE *)(*((_QWORD *)v23 + 1) + 8LL) == 14 )
    {
      LOWORD(v62) = sub_B2BD00((__int64)v23);
      v26 = 0;
      if ( BYTE1(v62) )
        v26 = v62;
      return sub_2A931D0(v105.m128i_i64, v26);
    }
    v35 = v103;
    v36 = *(_DWORD *)(v103 + 24);
    if ( v36 )
    {
      v37 = *(_QWORD *)(v103 + 8);
      v38 = 0;
      v39 = 1;
      v40 = (v36 - 1) & (((unsigned int)v104 >> 9) ^ ((unsigned int)v104 >> 4));
      v41 = (_BYTE *)(v37 + 16 * v40);
      v42 = *(_QWORD *)v41;
      if ( *(_QWORD *)v41 == v104 )
      {
LABEL_59:
        v41[8] = 0;
        return 0;
      }
      while ( v42 != -4096 )
      {
        if ( v42 == -8192 && !v38 )
          v38 = v41;
        LODWORD(v40) = (v36 - 1) & (v39 + v40);
        v41 = (_BYTE *)(v37 + 16LL * (unsigned int)v40);
        v42 = *(_QWORD *)v41;
        if ( v104 == *(_QWORD *)v41 )
          goto LABEL_59;
        ++v39;
      }
      if ( v38 )
        v41 = v38;
      v109 = (unsigned __int64)v41;
      v63 = *(_DWORD *)(v103 + 16);
      ++*(_QWORD *)v103;
      v64 = v63 + 1;
      if ( 4 * v64 < 3 * v36 )
      {
        if ( v36 - *(_DWORD *)(v35 + 20) - v64 > v36 >> 3 )
        {
LABEL_111:
          *(_DWORD *)(v35 + 16) = v64;
          v41 = (_BYTE *)v109;
          if ( *(_QWORD *)v109 != -4096 )
            --*(_DWORD *)(v35 + 20);
          v65 = v104;
          v41[8] = 0;
          *(_QWORD *)v41 = v65;
          goto LABEL_59;
        }
LABEL_116:
        sub_2A92FF0(v35, v36);
        sub_2A91CA0(v35, &v104, &v109);
        v64 = *(_DWORD *)(v35 + 16) + 1;
        goto LABEL_111;
      }
    }
    else
    {
      v109 = 0;
      ++*(_QWORD *)v103;
    }
    v36 *= 2;
    goto LABEL_116;
  }
  v57 = (unsigned int)sub_DFE150(a1[5]) >> 3;
  v58 = v57;
  _BitScanReverse64(&v57, v57);
  v59 = v57 ^ 0x3F;
  if ( !v58 )
    v59 = 64;
  v98 = sub_2A931D0(v105.m128i_i64, 63 - v59);
  sub_2240A30(&v109);
  return v98;
}
