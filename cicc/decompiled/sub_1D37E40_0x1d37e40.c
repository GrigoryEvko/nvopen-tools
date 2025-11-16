// Function: sub_1D37E40
// Address: 0x1d37e40
//
__int64 __fastcall sub_1D37E40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        const void **a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        unsigned __int8 a10)
{
  char v11; // r13
  __int64 *v12; // r12
  char v13; // bl
  const void **v14; // rdx
  char v15; // al
  int v16; // ebx
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD *v20; // r13
  __int64 v21; // r12
  void *v22; // rdi
  char v24; // al
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rcx
  unsigned int v32; // ebx
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned int v35; // eax
  unsigned int v36; // r15d
  unsigned int v37; // r15d
  unsigned int i; // r14d
  unsigned int v39; // ecx
  unsigned int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r9
  __int64 v44; // r8
  __int64 v45; // rdx
  __int64 *v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r10
  __int128 v49; // rdi
  __int64 v50; // rax
  void *v51; // rsi
  __int64 v52; // rcx
  unsigned __int8 *v53; // rsi
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rsi
  unsigned int v59; // eax
  unsigned int v60; // edx
  __int64 v61; // r14
  int v62; // r8d
  int v63; // r9d
  __int64 v64; // rdx
  __int64 v65; // rax
  int v66; // ebx
  int v67; // r12d
  __int64 v68; // r15
  int v69; // r15d
  void *v70; // r10
  __int64 v71; // r9
  _QWORD *v72; // rsi
  __int128 v73; // rax
  __int64 v74; // rax
  const __m128i *v75; // rdx
  __m128i *v76; // rax
  const void **v77; // rdx
  __int128 v78; // [rsp-10h] [rbp-1C0h]
  const void **v79; // [rsp+8h] [rbp-1A8h]
  unsigned __int8 v80; // [rsp+17h] [rbp-199h]
  __int64 v81; // [rsp+30h] [rbp-180h]
  __int64 v82; // [rsp+38h] [rbp-178h]
  __int64 v83; // [rsp+40h] [rbp-170h]
  __int64 v84; // [rsp+40h] [rbp-170h]
  void *v85; // [rsp+40h] [rbp-170h]
  __int64 v86; // [rsp+40h] [rbp-170h]
  unsigned __int8 v87; // [rsp+40h] [rbp-170h]
  __int64 v88; // [rsp+48h] [rbp-168h]
  unsigned int v89; // [rsp+50h] [rbp-160h]
  __int64 v90; // [rsp+50h] [rbp-160h]
  __int64 v91; // [rsp+58h] [rbp-158h]
  int v92; // [rsp+58h] [rbp-158h]
  __int64 v93; // [rsp+58h] [rbp-158h]
  unsigned __int8 v94; // [rsp+60h] [rbp-150h]
  __int64 v95; // [rsp+68h] [rbp-148h]
  __int64 v96; // [rsp+80h] [rbp-130h] BYREF
  const void **v97; // [rsp+88h] [rbp-128h]
  __int64 v98; // [rsp+90h] [rbp-120h] BYREF
  const void **v99; // [rsp+98h] [rbp-118h]
  __int64 v100; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v101; // [rsp+A8h] [rbp-108h]
  unsigned __int64 v102; // [rsp+B0h] [rbp-100h] BYREF
  unsigned int v103; // [rsp+B8h] [rbp-F8h]
  void *src; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v105; // [rsp+C8h] [rbp-E8h]
  _BYTE v106[32]; // [rsp+D0h] [rbp-E0h] BYREF
  _QWORD *v107; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v108; // [rsp+F8h] [rbp-B8h]
  _QWORD v109[22]; // [rsp+100h] [rbp-B0h] BYREF

  v11 = a4;
  v12 = (__int64 *)a1;
  v13 = a6;
  v95 = a2;
  v96 = a4;
  v97 = a5;
  v94 = a6;
  if ( (_BYTE)a4 )
  {
    if ( (unsigned __int8)(a4 - 14) <= 0x5Fu )
    {
      switch ( (char)a4 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          v24 = 3;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          v24 = 4;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          v24 = 5;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          v24 = 6;
          break;
        case 55:
          v24 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v24 = 8;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          v24 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v24 = 10;
          break;
        default:
          v24 = 2;
          break;
      }
      v99 = 0;
      LOBYTE(v98) = v24;
      v15 = a4;
      goto LABEL_18;
    }
    goto LABEL_3;
  }
  if ( !(unsigned __int8)sub_1F58D20(&v96) )
  {
LABEL_3:
    v14 = v97;
    v15 = v11;
    goto LABEL_4;
  }
  v11 = sub_1F596B0(&v96);
  v15 = v96;
LABEL_4:
  LOBYTE(v98) = v11;
  v99 = v14;
  if ( !v15 )
  {
    if ( !(unsigned __int8)sub_1F58D20(&v96) )
      goto LABEL_6;
    goto LABEL_19;
  }
LABEL_18:
  if ( (unsigned __int8)(v15 - 14) > 0x5Fu )
    goto LABEL_6;
LABEL_19:
  sub_1F40D10(&v107, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 48), v98, v99);
  if ( (_BYTE)v107 == 1 )
  {
    sub_1F40D10(&v107, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 48), v98, v99);
    LOBYTE(v98) = v108;
    v99 = (const void **)v109[0];
    v58 = a2 + 24;
    if ( (_BYTE)v108 )
    {
      v60 = sub_1D13440(v108);
    }
    else
    {
      v59 = sub_1F58D40(&v98, v58, v54, v55, v56, v57);
      v58 = v95 + 24;
      v60 = v59;
    }
    sub_16A5D10((__int64)&v107, v58, v60);
    v95 = sub_159C0E0(*(__int64 **)(a1 + 48), (__int64)&v107);
    if ( (unsigned int)v108 > 0x40 && v107 )
      j_j___libc_free_0_0(v107);
    goto LABEL_6;
  }
  if ( !*(_BYTE *)(a1 + 658) )
    goto LABEL_6;
  if ( !(_BYTE)v96 )
  {
    if ( (unsigned __int8)sub_1F58D20(&v96) )
      goto LABEL_23;
LABEL_6:
    v107 = v109;
    v16 = v13 == 0 ? 10 : 32;
    v108 = 0x2000000000LL;
    v91 = sub_1D29190(a1, v98, (__int64)v99, a4, (__int64)a5, a6);
    sub_16BD430((__int64)&v107, v16);
    sub_16BD4C0((__int64)&v107, v91);
    sub_16BD4C0((__int64)&v107, v95);
    sub_16BD430((__int64)&v107, a10);
    v102 = 0;
    v20 = sub_1D17920(a1, (__int64)&v107, a3, (__int64 *)&v102);
    if ( v20 )
    {
      if ( (_BYTE)v96 )
      {
        if ( (unsigned __int8)(v96 - 14) > 0x5Fu )
        {
LABEL_9:
          v21 = (__int64)v20;
          goto LABEL_10;
        }
      }
      else if ( !(unsigned __int8)sub_1F58D20(&v96) )
      {
        goto LABEL_9;
      }
    }
    else
    {
      v20 = *(_QWORD **)(a1 + 208);
      v47 = (unsigned __int8)v98;
      v48 = (__int64)v99;
      v92 = *(_DWORD *)(a3 + 8);
      if ( v20 )
      {
        *(_QWORD *)(a1 + 208) = *v20;
      }
      else
      {
        v90 = (__int64)v99;
        v87 = v98;
        v74 = sub_145CBF0((__int64 *)(a1 + 216), 112, 8);
        v47 = v87;
        v48 = v90;
        v20 = (_QWORD *)v74;
      }
      *((_QWORD *)&v49 + 1) = v48;
      *(_QWORD *)&v49 = (unsigned __int8)v47;
      v50 = sub_1D274F0(v49, v17, v47, v18, v19);
      v51 = *(void **)a3;
      v52 = v50;
      src = v51;
      if ( v51 )
      {
        v84 = v50;
        sub_1623A60((__int64)&src, (__int64)v51, 2);
        v52 = v84;
      }
      *v20 = 0;
      v20[7] = 0x100000000LL;
      v20[1] = 0;
      v20[2] = 0;
      *((_WORD *)v20 + 12) = v16;
      *((_DWORD *)v20 + 7) = -1;
      v20[4] = 0;
      v20[5] = v52;
      v20[6] = 0;
      *((_DWORD *)v20 + 16) = v92;
      v53 = (unsigned __int8 *)src;
      v20[9] = src;
      if ( v53 )
        sub_1623210((__int64)&src, v53, (__int64)(v20 + 9));
      *((_WORD *)v20 + 40) &= 0xF000u;
      *((_WORD *)v20 + 13) = 0;
      *((_BYTE *)v20 + 26) = 8 * (a10 & 1);
      v20[11] = v95;
      sub_16BDA20(v12 + 40, v20, (__int64 *)v102);
      sub_1D172A0((__int64)v12, (__int64)v20);
      if ( (_BYTE)v96 )
      {
        if ( (unsigned __int8)(v96 - 14) > 0x5Fu )
          goto LABEL_15;
      }
      else if ( !(unsigned __int8)sub_1F58D20(&v96) )
      {
        goto LABEL_15;
      }
    }
    v20 = sub_1D35F20(v12, (unsigned int)v96, v97, a3, (__int64)v20, 0, *(double *)a7.m128i_i64, a8, a9);
LABEL_15:
    v21 = (__int64)v20;
LABEL_10:
    v22 = v107;
    if ( v107 == v109 )
      return v21;
LABEL_11:
    _libc_free((unsigned __int64)v22);
    return v21;
  }
  if ( (unsigned __int8)(v96 - 14) > 0x5Fu )
    goto LABEL_6;
LABEL_23:
  sub_1F40D10(&v107, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 48), v98, v99);
  if ( (_BYTE)v107 != 2 )
    goto LABEL_6;
  v25 = *(_QWORD *)(a1 + 16);
  sub_1F40D10(&v107, v25, *(_QWORD *)(a1 + 48), v98, v99);
  LOBYTE(v100) = v108;
  v101 = v109[0];
  if ( (_BYTE)v108 )
    v32 = sub_1D13440(v108);
  else
    v32 = sub_1F58D40(&v100, v25, v26, v27, v28, v29);
  if ( (_BYTE)v96 )
    v35 = sub_1D13440(v96);
  else
    v35 = sub_1F58D40(&v96, v25, v30, v31, v33, v34);
  v36 = v100;
  v83 = *(_QWORD *)(a1 + 48);
  v82 = v101;
  v89 = v35 / v32;
  v79 = 0;
  v80 = sub_1D15020(v100, v35 / v32);
  if ( !v80 )
  {
    v80 = sub_1F593D0(v83, v36, v82, v89);
    v79 = v77;
  }
  v81 = a3;
  v37 = 0;
  src = v106;
  v105 = 0x200000000LL;
  for ( i = 0; ; ++i )
  {
    v39 = (_BYTE)v96 ? word_42E7700[(unsigned __int8)(v96 - 14)] : sub_1F58D30(&v96);
    if ( v89 / v39 <= i )
      break;
    v40 = *(_DWORD *)(v95 + 32);
    v103 = v40;
    if ( v40 > 0x40 )
    {
      sub_16A4FD0((__int64)&v102, (const void **)(v95 + 24));
      v40 = v103;
      if ( v103 > 0x40 )
      {
        sub_16A8110((__int64)&v102, v37);
        goto LABEL_37;
      }
    }
    else
    {
      v102 = *(_QWORD *)(v95 + 24);
    }
    if ( v37 == v40 )
      v102 = 0;
    else
      v102 >>= v37;
LABEL_37:
    sub_16A5D10((__int64)&v107, (__int64)&v102, v32);
    v41 = sub_1D38970(a1, (unsigned int)&v107, v81, v100, v101, v94, a10);
    v43 = v42;
    v44 = v41;
    v45 = (unsigned int)v105;
    if ( (unsigned int)v105 >= HIDWORD(v105) )
    {
      v86 = v41;
      v88 = v43;
      sub_16CD150((__int64)&src, v106, 0, 16, v41, v43);
      v45 = (unsigned int)v105;
      v44 = v86;
      v43 = v88;
    }
    v46 = (__int64 *)((char *)src + 16 * v45);
    *v46 = v44;
    v46[1] = v43;
    LODWORD(v105) = v105 + 1;
    if ( (unsigned int)v108 > 0x40 && v107 )
      j_j___libc_free_0_0(v107);
    if ( v103 > 0x40 && v102 )
      j_j___libc_free_0_0(v102);
    v37 += v32;
  }
  v61 = v81;
  if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(a1 + 32)) )
  {
    v75 = (const __m128i *)src;
    v76 = (__m128i *)((char *)src + 16 * (unsigned int)v105);
    if ( v76 != src )
    {
      while ( v75 < --v76 )
      {
        a7 = _mm_loadu_si128(v75++);
        v75[-1].m128i_i64[0] = v76->m128i_i64[0];
        v75[-1].m128i_i32[2] = v76->m128i_i32[2];
        v76->m128i_i64[0] = a7.m128i_i64[0];
        v76->m128i_i32[2] = a7.m128i_i32[2];
      }
    }
  }
  v107 = v109;
  v108 = 0x800000000LL;
  if ( (_BYTE)v96 )
    v63 = word_42E7700[(unsigned __int8)(v96 - 14)];
  else
    v63 = sub_1F58D30(&v96);
  if ( v63 )
  {
    v64 = 8;
    v65 = 0;
    v66 = 0;
    v67 = v63;
    while ( 1 )
    {
      v69 = v105;
      v70 = src;
      v71 = 16LL * (unsigned int)v105;
      if ( (unsigned int)v105 > (unsigned __int64)(v64 - v65) )
      {
        v85 = src;
        v93 = 16LL * (unsigned int)v105;
        sub_16CD150((__int64)&v107, v109, v65 + (unsigned int)v105, 16, v62, v71);
        v65 = (unsigned int)v108;
        v70 = v85;
        v71 = v93;
      }
      if ( v71 )
      {
        memcpy(&v107[2 * v65], v70, v71);
        LODWORD(v65) = v108;
      }
      LODWORD(v68) = v65 + v69;
      ++v66;
      LODWORD(v108) = v68;
      v65 = (unsigned int)v68;
      if ( v66 == v67 )
        break;
      v64 = HIDWORD(v108);
    }
    v72 = v107;
    v12 = (__int64 *)a1;
    v68 = (unsigned int)v68;
    v61 = v81;
  }
  else
  {
    v72 = v109;
    v68 = 0;
  }
  *((_QWORD *)&v78 + 1) = v68;
  *(_QWORD *)&v78 = v72;
  *(_QWORD *)&v73 = sub_1D359D0(v12, 104, v61, v80, v79, 0, *(double *)a7.m128i_i64, a8, a9, v78);
  v21 = sub_1D309E0(v12, 158, v61, (unsigned int)v96, v97, 0, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, v73);
  if ( v107 != v109 )
    _libc_free((unsigned __int64)v107);
  v22 = src;
  if ( src != v106 )
    goto LABEL_11;
  return v21;
}
