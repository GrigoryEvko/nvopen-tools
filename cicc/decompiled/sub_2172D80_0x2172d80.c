// Function: sub_2172D80
// Address: 0x2172d80
//
void __fastcall sub_2172D80(__int64 a1, __int64 *a2, __int64 a3, __m128i a4, double a5, __m128i a6)
{
  unsigned __int64 v6; // rbx
  __int64 v8; // rax
  __int8 v9; // dl
  const void **v10; // r15
  char v11; // si
  unsigned int v12; // eax
  int v13; // r9d
  __int8 v14; // dl
  unsigned __int8 v15; // dl
  __int64 v16; // r8
  __m128 *v17; // rax
  unsigned __int64 v18; // r13
  __m128 *v19; // rdx
  unsigned __int8 *v20; // r8
  __int64 v21; // rsi
  __int64 *v22; // rcx
  __m128i v23; // xmm2
  __int64 v24; // rsi
  __m128i v25; // xmm1
  __int64 v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // rax
  __m128i v29; // xmm4
  __int64 v30; // rax
  __int32 v31; // edx
  __m128i v32; // xmm3
  int v33; // r8d
  int v34; // r9d
  __m128i *v35; // rax
  __m128i v36; // xmm1
  __m128i si128; // xmm2
  __int64 v38; // rax
  __int64 v39; // r15
  __int64 v40; // rax
  int v41; // edx
  unsigned __int8 v42; // r10
  __int64 v43; // r11
  __int64 v44; // r13
  int v45; // r8d
  __int64 v46; // r13
  __int64 v47; // r14
  unsigned int v48; // r13d
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r9
  __int64 v52; // r8
  __int64 v53; // rdx
  __int64 *v54; // rdx
  __int64 v55; // rax
  __m128i *v56; // rdx
  __int64 v57; // rsi
  __m128i *v58; // r14
  __int64 v59; // r15
  int v60; // r8d
  int v61; // r9d
  __int64 *v62; // r14
  __int64 *v63; // rdx
  __int64 *v64; // r15
  __int64 v65; // rax
  __int64 **v66; // rax
  __int64 v67; // rsi
  __int64 v68; // rax
  __int64 v69; // rbx
  __int64 *v70; // rax
  __m128i *v71; // rdi
  int v72; // r8d
  int v73; // r9d
  __int64 v74; // r14
  __int64 v75; // rdx
  __int64 v76; // r15
  __int64 v77; // rax
  __int64 *v78; // rax
  __int64 v79; // rax
  __int64 *v80; // rax
  __m128i v81; // xmm3
  char v82; // al
  const void **v83; // rdx
  __int128 v84; // [rsp-10h] [rbp-290h]
  int v85; // [rsp+8h] [rbp-278h]
  unsigned __int8 v86; // [rsp+10h] [rbp-270h]
  __int64 v87; // [rsp+18h] [rbp-268h]
  unsigned __int16 v88; // [rsp+20h] [rbp-260h]
  __int64 v89; // [rsp+20h] [rbp-260h]
  __int64 v90; // [rsp+28h] [rbp-258h]
  __int8 v91; // [rsp+40h] [rbp-240h]
  __int64 *v92; // [rsp+40h] [rbp-240h]
  unsigned int v93; // [rsp+40h] [rbp-240h]
  __int64 v94; // [rsp+48h] [rbp-238h]
  __int8 v95; // [rsp+50h] [rbp-230h]
  unsigned int v98; // [rsp+70h] [rbp-210h] BYREF
  const void **v99; // [rsp+78h] [rbp-208h]
  __m128i v100; // [rsp+80h] [rbp-200h] BYREF
  __int64 v101; // [rsp+90h] [rbp-1F0h] BYREF
  int v102; // [rsp+98h] [rbp-1E8h]
  __int64 v103[4]; // [rsp+A0h] [rbp-1E0h] BYREF
  __m128i v104; // [rsp+C0h] [rbp-1C0h] BYREF
  __m128i v105; // [rsp+D0h] [rbp-1B0h] BYREF
  __int64 *v106; // [rsp+E0h] [rbp-1A0h] BYREF
  __int64 v107; // [rsp+E8h] [rbp-198h]
  _BYTE v108[64]; // [rsp+F0h] [rbp-190h] BYREF
  __m128 *v109; // [rsp+130h] [rbp-150h] BYREF
  __int64 v110; // [rsp+138h] [rbp-148h]
  _OWORD v111[8]; // [rsp+140h] [rbp-140h] BYREF
  __m128i v112; // [rsp+1C0h] [rbp-C0h] BYREF
  __m128i v113; // [rsp+1D0h] [rbp-B0h] BYREF
  __m128i v114; // [rsp+1E0h] [rbp-A0h] BYREF
  __m128i v115; // [rsp+1F0h] [rbp-90h] BYREF
  __m128i v116[8]; // [rsp+200h] [rbp-80h] BYREF

  v8 = *(_QWORD *)(a1 + 40);
  v9 = *(_BYTE *)v8;
  v10 = *(const void ***)(v8 + 8);
  v112.m128i_i8[0] = v9;
  v112.m128i_i64[1] = (__int64)v10;
  if ( v9 )
  {
    if ( (unsigned __int8)(v9 - 14) > 0x5Fu )
    {
      LOBYTE(v98) = v9;
      v11 = v9;
      v99 = v10;
      v88 = 678;
    }
    else
    {
      if ( word_432BB60[(unsigned __int8)(v9 - 14)] == 2 )
      {
        LOBYTE(v98) = v9;
        v99 = v10;
        v88 = 679;
      }
      else
      {
        LOBYTE(v98) = v9;
        v99 = v10;
        v88 = 680;
      }
      switch ( v9 )
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
          v9 = v98;
          v10 = 0;
          v11 = 3;
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
          v9 = v98;
          v10 = 0;
          v11 = 4;
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
          v9 = v98;
          v10 = 0;
          v11 = 5;
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
          v9 = v98;
          v10 = 0;
          v11 = 6;
          break;
        case 55:
          v9 = v98;
          v10 = 0;
          v11 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v9 = v98;
          v10 = 0;
          v11 = 8;
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
          v9 = v98;
          v10 = 0;
          v11 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v9 = v98;
          v10 = 0;
          v11 = 10;
          break;
        default:
          v9 = v98;
          v10 = 0;
          v11 = 2;
          break;
      }
    }
    goto LABEL_10;
  }
  if ( sub_1F58D20((__int64)&v112) )
  {
    LOBYTE(v98) = 0;
    v99 = v10;
    if ( (unsigned int)sub_1F58D30((__int64)&v112) == 2 )
      v88 = 679;
    else
      v88 = 680;
  }
  else
  {
    LOBYTE(v98) = 0;
    v99 = v10;
    v88 = 678;
  }
  if ( sub_1F58D20((__int64)&v98) )
  {
    v82 = sub_1F596B0((__int64)&v98);
    v10 = v83;
    v11 = v82;
  }
  else
  {
    v11 = v98;
    v10 = v99;
  }
  v9 = v98;
  v112.m128i_i8[0] = v11;
  v112.m128i_i64[1] = (__int64)v10;
  if ( v11 )
  {
LABEL_10:
    v91 = v9;
    v12 = sub_216FFF0(v11);
    v14 = v91;
    goto LABEL_11;
  }
  v95 = v98;
  v12 = sub_1F58D40((__int64)&v112);
  v14 = v95;
  v11 = 0;
LABEL_11:
  v103[1] = (__int64)&v98;
  if ( v12 <= 0xF )
    v10 = 0;
  v103[3] = a1;
  if ( v12 <= 0xF )
    v11 = 4;
  v103[0] = (__int64)&v100;
  v100.m128i_i8[0] = v11;
  v103[2] = (__int64)a2;
  v109 = (__m128 *)v111;
  v100.m128i_i64[1] = (__int64)v10;
  v110 = 0x800000000LL;
  if ( !v14 )
  {
    if ( sub_1F58D20((__int64)&v98) )
    {
      v16 = (unsigned int)sub_1F58D30((__int64)&v98);
      v17 = (__m128 *)v111;
      v18 = (unsigned int)v16;
      if ( (unsigned int)v16 <= 8 )
        goto LABEL_18;
LABEL_62:
      v93 = v16;
      sub_16CD150((__int64)&v109, v111, v18, 16, v16, v13);
      v17 = v109;
      v16 = v93;
      goto LABEL_18;
    }
LABEL_69:
    v81 = _mm_load_si128(&v100);
    LODWORD(v110) = 1;
    v16 = 1;
    v111[0] = v81;
    goto LABEL_25;
  }
  v15 = v14 - 14;
  if ( v15 > 0x5Fu )
    goto LABEL_69;
  v16 = word_432BB60[v15];
  v17 = (__m128 *)v111;
  v18 = v16;
  if ( (unsigned int)v16 > 8 )
    goto LABEL_62;
LABEL_18:
  LODWORD(v110) = v16;
  v19 = &v17[v18];
  if ( v19 != v17 )
  {
    do
    {
      if ( v17 )
      {
        a4 = _mm_load_si128(&v100);
        *v17 = (__m128)a4;
      }
      ++v17;
    }
    while ( v19 != v17 );
    v16 = (unsigned int)v110;
  }
  if ( (unsigned int)v16 >= HIDWORD(v110) )
  {
    sub_16CD150((__int64)&v109, v111, 0, 16, v16, v13);
    v16 = (unsigned int)v110;
  }
LABEL_25:
  v20 = (unsigned __int8 *)&v109[v16];
  *(_QWORD *)v20 = 1;
  *((_QWORD *)v20 + 1) = 0;
  v21 = *(_QWORD *)(a1 + 72);
  LODWORD(v110) = v110 + 1;
  v112.m128i_i64[0] = v21;
  if ( v21 )
    sub_1623A60((__int64)&v112, v21, 2);
  v22 = *(__int64 **)(a1 + 32);
  v112.m128i_i32[2] = *(_DWORD *)(a1 + 64);
  sub_2170130((__int64)&v104, v22[5], v22[10], v22[11], v22[20], v22[21], a4, a5, a6, (__int64)&v112, a2);
  if ( v112.m128i_i64[0] )
    sub_161E7C0((__int64)&v112, v112.m128i_i64[0]);
  v23 = _mm_load_si128(&v104);
  v24 = *(_QWORD *)(a1 + 72);
  v25 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 32));
  v113 = v23;
  v101 = v24;
  v112 = v25;
  if ( v24 )
    sub_1623A60((__int64)&v101, v24, 2);
  v26 = *(_QWORD *)(a1 + 104);
  v102 = *(_DWORD *)(a1 + 64);
  v27 = sub_1E340A0(v26);
  v28 = sub_1D38BB0((__int64)a2, v27, (__int64)&v101, 3, 0, 1, a4, *(double *)v25.m128i_i64, v23, 0);
  v29 = _mm_load_si128(&v105);
  v114.m128i_i64[0] = v28;
  v30 = *(_QWORD *)(a1 + 32);
  v114.m128i_i32[2] = v31;
  v32 = _mm_loadu_si128((const __m128i *)(v30 + 120));
  v106 = (__int64 *)v108;
  v107 = 0x400000000LL;
  v115 = v32;
  v116[0] = v29;
  sub_16CD150((__int64)&v106, v108, 5u, 16, v33, v34);
  v35 = (__m128i *)&v106[2 * (unsigned int)v107];
  *v35 = _mm_load_si128(&v112);
  v35[1] = _mm_load_si128(&v113);
  v35[2] = _mm_load_si128(&v114);
  v36 = _mm_load_si128(&v115);
  v35[3] = v36;
  si128 = _mm_load_si128(v116);
  v35[4] = si128;
  v38 = (unsigned int)(v107 + 5);
  LODWORD(v107) = v107 + 5;
  if ( v101 )
  {
    sub_161E7C0((__int64)&v101, v101);
    v38 = (unsigned int)v107;
  }
  v94 = v38;
  v92 = v106;
  v86 = *(_BYTE *)(a1 + 88);
  v87 = *(_QWORD *)(a1 + 96);
  v39 = *(_QWORD *)(a1 + 104);
  v40 = sub_1D25C30((__int64)a2, (unsigned __int8 *)v109, (unsigned int)v110);
  v42 = v86;
  v43 = v87;
  v44 = v40;
  v45 = v41;
  v112.m128i_i64[0] = *(_QWORD *)(a1 + 72);
  if ( v112.m128i_i64[0] )
  {
    v85 = v41;
    sub_1623A60((__int64)&v112, v112.m128i_i64[0], 2);
    v45 = v85;
    v42 = v86;
    v43 = v87;
  }
  v112.m128i_i32[2] = *(_DWORD *)(a1 + 64);
  v46 = sub_1D24DC0(a2, v88, (__int64)&v112, v44, v45, v39, v92, v94, v42, v43);
  if ( v112.m128i_i64[0] )
    sub_161E7C0((__int64)&v112, v112.m128i_i64[0]);
  if ( (_BYTE)v98 )
  {
    if ( (unsigned __int8)(v98 - 14) <= 0x5Fu )
    {
LABEL_39:
      v112.m128i_i64[0] = (__int64)&v113;
      v112.m128i_i64[1] = 0x800000000LL;
      if ( *(_DWORD *)(v46 + 60) == 1 )
      {
        v56 = &v113;
        v55 = 0;
      }
      else
      {
        v47 = v46;
        v48 = 0;
        do
        {
          v6 = v48 | v6 & 0xFFFFFFFF00000000LL;
          v49 = sub_21703B0(
                  v103,
                  v47,
                  v6,
                  *(double *)a4.m128i_i64,
                  *(double *)v36.m128i_i64,
                  *(double *)si128.m128i_i64);
          v51 = v50;
          v52 = v49;
          v53 = v112.m128i_u32[2];
          if ( v112.m128i_i32[2] >= (unsigned __int32)v112.m128i_i32[3] )
          {
            v89 = v49;
            v90 = v51;
            sub_16CD150((__int64)&v112, &v113, 0, 16, v49, v51);
            v53 = v112.m128i_u32[2];
            v52 = v89;
            v51 = v90;
          }
          v54 = (__int64 *)(v112.m128i_i64[0] + 16 * v53);
          ++v48;
          *v54 = v52;
          v54[1] = v51;
          v55 = (unsigned int)++v112.m128i_i32[2];
        }
        while ( *(_DWORD *)(v47 + 60) - 1 > v48 );
        v56 = (__m128i *)v112.m128i_i64[0];
        v46 = v47;
      }
      v57 = *(_QWORD *)(a1 + 72);
      v58 = v56;
      v59 = v55;
      v101 = v57;
      if ( v57 )
        sub_1623A60((__int64)&v101, v57, 2);
      *((_QWORD *)&v84 + 1) = v59;
      *(_QWORD *)&v84 = v58;
      v102 = *(_DWORD *)(a1 + 64);
      v62 = sub_1D359D0(
              a2,
              104,
              (__int64)&v101,
              v98,
              v99,
              0,
              *(double *)a4.m128i_i64,
              *(double *)v36.m128i_i64,
              si128,
              v84);
      v64 = v63;
      v65 = *(unsigned int *)(a3 + 8);
      if ( (unsigned int)v65 >= *(_DWORD *)(a3 + 12) )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v60, v61);
        v65 = *(unsigned int *)(a3 + 8);
      }
      v66 = (__int64 **)(*(_QWORD *)a3 + 16 * v65);
      *v66 = v62;
      v66[1] = v64;
      v67 = v101;
      v68 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      *(_DWORD *)(a3 + 8) = v68;
      if ( v67 )
      {
        sub_161E7C0((__int64)&v101, v67);
        v68 = *(unsigned int *)(a3 + 8);
      }
      v69 = (unsigned int)(*(_DWORD *)(v46 + 60) - 1);
      if ( (unsigned int)v68 >= *(_DWORD *)(a3 + 12) )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v60, v61);
        v68 = *(unsigned int *)(a3 + 8);
      }
      v70 = (__int64 *)(*(_QWORD *)a3 + 16 * v68);
      *v70 = v46;
      v70[1] = v69;
      v71 = (__m128i *)v112.m128i_i64[0];
      ++*(_DWORD *)(a3 + 8);
      if ( v71 != &v113 )
        _libc_free((unsigned __int64)v71);
      goto LABEL_55;
    }
  }
  else if ( sub_1F58D20((__int64)&v98) )
  {
    goto LABEL_39;
  }
  v74 = sub_21703B0(v103, v46, 0, *(double *)a4.m128i_i64, *(double *)v36.m128i_i64, *(double *)si128.m128i_i64);
  v76 = v75;
  v77 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v77 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v72, v73);
    v77 = *(unsigned int *)(a3 + 8);
  }
  v78 = (__int64 *)(*(_QWORD *)a3 + 16 * v77);
  *v78 = v74;
  v78[1] = v76;
  v79 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v79;
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v79 )
  {
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, v72, v73);
    v79 = *(unsigned int *)(a3 + 8);
  }
  v80 = (__int64 *)(*(_QWORD *)a3 + 16 * v79);
  *v80 = v46;
  v80[1] = 1;
  ++*(_DWORD *)(a3 + 8);
LABEL_55:
  if ( v106 != (__int64 *)v108 )
    _libc_free((unsigned __int64)v106);
  if ( v109 != (__m128 *)v111 )
    _libc_free((unsigned __int64)v109);
}
