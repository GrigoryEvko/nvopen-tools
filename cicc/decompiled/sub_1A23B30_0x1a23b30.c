// Function: sub_1A23B30
// Address: 0x1a23b30
//
__int64 __fastcall sub_1A23B30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 v9; // r12
  unsigned int v10; // eax
  __int64 v11; // r14
  unsigned __int8 v12; // dl
  __m128i v13; // xmm0
  __int64 v14; // rbx
  __int64 v15; // r15
  unsigned __int64 v16; // rax
  __int64 v17; // r15
  int v18; // eax
  int v19; // edx
  __int64 **v20; // rax
  __int64 *v21; // r14
  int v22; // ebx
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  __int64 *v26; // r12
  char v27; // dl
  unsigned int v28; // ebx
  unsigned int v29; // ecx
  __int64 *v30; // r12
  char v31; // dl
  unsigned int v33; // ebx
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // r15
  _QWORD *v37; // rax
  _QWORD *v38; // rax
  _QWORD *v39; // rax
  _QWORD *v40; // r12
  __int64 v41; // r14
  __int64 v42; // rax
  __int64 *v43; // rax
  __int64 *v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rsi
  unsigned int v48; // eax
  _QWORD *v49; // rax
  __int64 v50; // rax
  _BYTE *v51; // [rsp+8h] [rbp-188h]
  __int64 v52; // [rsp+10h] [rbp-180h]
  __int64 v53; // [rsp+30h] [rbp-160h]
  int v56; // [rsp+48h] [rbp-148h]
  _QWORD *v57; // [rsp+50h] [rbp-140h] BYREF
  unsigned int v58; // [rsp+58h] [rbp-138h]
  __int64 v59; // [rsp+60h] [rbp-130h] BYREF
  unsigned int v60; // [rsp+68h] [rbp-128h]
  __int64 *v61; // [rsp+70h] [rbp-120h] BYREF
  unsigned int v62; // [rsp+78h] [rbp-118h]
  __m128i v63; // [rsp+80h] [rbp-110h] BYREF
  char v64; // [rsp+90h] [rbp-100h]
  char v65; // [rsp+91h] [rbp-FFh]
  __m128i v66[2]; // [rsp+A0h] [rbp-F0h] BYREF
  __m128i v67; // [rsp+C0h] [rbp-D0h] BYREF
  __int64 v68; // [rsp+D0h] [rbp-C0h]
  _BYTE *v69; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v70; // [rsp+E8h] [rbp-A8h]
  _BYTE v71[32]; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v72; // [rsp+110h] [rbp-80h] BYREF
  _BYTE *v73; // [rsp+118h] [rbp-78h]
  _BYTE *v74; // [rsp+120h] [rbp-70h]
  __int64 v75; // [rsp+128h] [rbp-68h]
  int v76; // [rsp+130h] [rbp-60h]
  _BYTE v77[88]; // [rsp+138h] [rbp-58h] BYREF

  v9 = a3;
  v72 = 0;
  v73 = v77;
  v74 = v77;
  v75 = 4;
  v76 = 0;
  sub_1412190((__int64)&v72, a3);
  v69 = v71;
  v70 = 0x400000000LL;
  v10 = *(_DWORD *)(a4 + 8);
  v58 = v10;
  if ( v10 > 0x40 )
  {
    sub_16A4EF0((__int64)&v57, 0, 0);
    v10 = *(_DWORD *)(a4 + 8);
  }
  else
  {
    v57 = 0;
  }
  v11 = 0;
  v51 = 0;
  v52 = **(_QWORD **)(a5 + 16);
  while ( 1 )
  {
    v12 = *(_BYTE *)(v9 + 16);
    if ( v12 <= 0x17u )
    {
      if ( v12 != 5 || *(_WORD *)(v9 + 18) != 32 )
        goto LABEL_6;
    }
    else if ( v12 != 56 )
    {
      goto LABEL_6;
    }
    v67.m128i_i32[2] = v10;
    if ( v10 > 0x40 )
      sub_16A4EF0((__int64)&v67, 0, 0);
    else
      v67.m128i_i64[0] = 0;
    if ( (unsigned __int8)sub_1634900(v9, a2, (__int64)&v67) )
    {
      sub_16A7200(a4, v67.m128i_i64);
      v26 = (*(_BYTE *)(v9 + 23) & 0x40) != 0
          ? *(__int64 **)(v9 - 8)
          : (__int64 *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      v9 = *v26;
      sub_1412190((__int64)&v72, v9);
      if ( v27 )
      {
        v17 = v11;
        sub_135E100(v67.m128i_i64);
        goto LABEL_75;
      }
    }
    if ( v67.m128i_i32[2] > 0x40u && v67.m128i_i64[0] )
      j_j___libc_free_0_0(v67.m128i_i64[0]);
    v10 = *(_DWORD *)(a4 + 8);
LABEL_6:
    LODWORD(v70) = 0;
    v60 = v10;
    if ( v10 > 0x40 )
      sub_16A4FD0((__int64)&v59, (const void **)a4);
    else
      v59 = *(_QWORD *)a4;
    v13 = _mm_loadu_si128((const __m128i *)&a7);
    v68 = a8;
    v67 = v13;
    v14 = *(_QWORD *)v9;
    if ( v14 == sub_16471D0(*(_QWORD **)(a1 + 24), *(_DWORD *)(*(_QWORD *)v9 + 8LL) >> 8) && sub_1642F90(v52, 8)
      || ((v15 = *(_QWORD *)(v14 + 24), v16 = *(unsigned __int8 *)(v15 + 8), (unsigned __int8)v16 > 0xFu)
       || (v24 = 35454, !_bittest64(&v24, v16)))
      && ((unsigned int)(v16 - 13) > 1 && (_DWORD)v16 != 16 || !sub_16435F0(*(_QWORD *)(v14 + 24), 0)) )
    {
LABEL_13:
      if ( v60 <= 0x40 )
        goto LABEL_64;
LABEL_14:
      v17 = 0;
LABEL_15:
      if ( v59 )
        j_j___libc_free_0_0(v59);
      goto LABEL_17;
    }
    v25 = sub_12BE0A0(a2, v15);
    v62 = v60;
    if ( v60 <= 0x40 )
    {
      v61 = (__int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v60) & v25);
LABEL_37:
      if ( !v61 )
        goto LABEL_13;
      goto LABEL_38;
    }
    sub_16A4EF0((__int64)&v61, v25, 0);
    v28 = v62;
    if ( v62 <= 0x40 )
      goto LABEL_37;
    if ( v28 - (unsigned int)sub_16A57B0((__int64)&v61) <= 0x40 && !*v61 )
    {
      j_j___libc_free_0_0(v61);
      if ( v60 <= 0x40 )
      {
LABEL_64:
        v17 = v11;
        goto LABEL_24;
      }
      goto LABEL_14;
    }
LABEL_38:
    sub_16A9F90((__int64)&v63, (__int64)&v59, (__int64)&v61);
    sub_16A7B50((__int64)v66, (__int64)&v63, (__int64 *)&v61);
    sub_16A7590((__int64)&v59, v66[0].m128i_i64);
    if ( v66[0].m128i_i32[2] > 0x40u && v66[0].m128i_i64[0] )
      j_j___libc_free_0_0(v66[0].m128i_i64[0]);
    v66[0].m128i_i64[0] = sub_159C0E0(*(__int64 **)(a1 + 24), (__int64)&v63);
    sub_12A9700((__int64)&v69, v66);
    v17 = sub_1A234E0(a1, a2, (__int64 *)v9, v15, (__int64)&v59, v52, v13, (__int64)&v69, v67.m128i_i64[0]);
    if ( v63.m128i_i32[2] > 0x40u && v63.m128i_i64[0] )
      j_j___libc_free_0_0(v63.m128i_i64[0]);
    if ( v62 > 0x40 && v61 )
      j_j___libc_free_0_0(v61);
    if ( v60 > 0x40 )
      goto LABEL_15;
LABEL_17:
    if ( !v17 )
      goto LABEL_64;
    if ( v11 && v11 != v53 && *(_BYTE *)(v11 + 16) > 0x17u )
      sub_15F20C0((_QWORD *)v11);
    if ( a5 == *(_QWORD *)v17 )
      goto LABEL_85;
    v53 = v9;
LABEL_24:
    if ( !sub_1642F90(*(_QWORD *)v9, 8) )
      goto LABEL_25;
    if ( v58 <= 0x40 )
    {
      v29 = *(_DWORD *)(a4 + 8);
      if ( v29 <= 0x40 )
      {
        v58 = *(_DWORD *)(a4 + 8);
        v51 = (_BYTE *)v9;
        v57 = (_QWORD *)(*(_QWORD *)a4 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v29));
LABEL_25:
        v18 = *(unsigned __int8 *)(v9 + 16);
        if ( (unsigned __int8)v18 > 0x17u )
          goto LABEL_26;
        goto LABEL_69;
      }
    }
    sub_16A51C0((__int64)&v57, a4);
    v18 = *(unsigned __int8 *)(v9 + 16);
    v51 = (_BYTE *)v9;
    if ( (unsigned __int8)v18 > 0x17u )
    {
LABEL_26:
      v19 = v18 - 24;
      if ( v18 != 71 )
        goto LABEL_27;
      goto LABEL_71;
    }
LABEL_69:
    if ( (_BYTE)v18 != 5 )
      break;
    v19 = *(unsigned __int16 *)(v9 + 18);
    if ( (_WORD)v19 != 47 )
    {
LABEL_27:
      if ( v19 != 48 )
        goto LABEL_80;
      if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
        v20 = *(__int64 ***)(v9 - 8);
      else
        v20 = (__int64 **)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
      v21 = *v20;
      v23 = **v20;
      if ( *(_BYTE *)(v23 + 8) == 16 )
        v23 = **(_QWORD **)(v23 + 16);
      v22 = *(_DWORD *)(a4 + 8);
      if ( 8 * (unsigned int)sub_15A95A0(a2, *(_DWORD *)(v23 + 8) >> 8) != v22 )
        goto LABEL_80;
      v9 = (__int64)v21;
      goto LABEL_74;
    }
LABEL_71:
    if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
      v30 = *(__int64 **)(v9 - 8);
    else
      v30 = (__int64 *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF));
    v9 = *v30;
LABEL_74:
    sub_1412190((__int64)&v72, v9);
    if ( !v31 )
      goto LABEL_80;
LABEL_75:
    v10 = *(_DWORD *)(a4 + 8);
    v11 = v17;
  }
  if ( (_BYTE)v18 == 1 )
    __asm { jmp     rax }
LABEL_80:
  if ( !v17 )
  {
    if ( !v51 )
    {
      v65 = 1;
      v64 = 3;
      v63.m128i_i64[0] = (__int64)"sroa_raw_cast";
      sub_14EC200(v66, (const __m128i *)&a7, &v63);
      v46 = a5;
      if ( *(_BYTE *)(a5 + 8) == 16 )
        v46 = **(_QWORD **)(a5 + 16);
      v47 = sub_16471D0(*(_QWORD **)(a1 + 24), *(_DWORD *)(v46 + 8) >> 8);
      if ( v47 == *(_QWORD *)v9 )
      {
        v51 = (_BYTE *)v9;
      }
      else if ( *(_BYTE *)(v9 + 16) > 0x10u )
      {
        LOWORD(v68) = 257;
        v49 = (_QWORD *)sub_15FDF90(v9, v47, (__int64)&v67, 0);
        v51 = sub_1A1C7B0((__int64 *)a1, v49, v66);
      }
      else
      {
        v51 = (_BYTE *)sub_15A4AD0((__int64 ***)v9, v47);
      }
      if ( v58 <= 0x40 )
      {
        v48 = *(_DWORD *)(a4 + 8);
        if ( v48 <= 0x40 )
        {
          v58 = *(_DWORD *)(a4 + 8);
          v57 = (_QWORD *)(*(_QWORD *)a4 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v48));
          goto LABEL_108;
        }
      }
      sub_16A51C0((__int64)&v57, a4);
    }
    v33 = v58;
    if ( v58 > 0x40 )
    {
      if ( v33 - (unsigned int)sub_16A57B0((__int64)&v57) > 0x40 )
        goto LABEL_103;
      v37 = (_QWORD *)*v57;
LABEL_109:
      v17 = (__int64)v51;
      if ( !v37 )
        goto LABEL_81;
LABEL_103:
      v65 = 1;
      v64 = 3;
      v63.m128i_i64[0] = (__int64)"sroa_raw_idx";
      sub_14EC200(v66, (const __m128i *)&a7, &v63);
      v34 = sub_159C0E0(*(__int64 **)(a1 + 24), (__int64)&v57);
      v35 = sub_1643330(*(_QWORD **)(a1 + 24));
      v59 = v34;
      v36 = v35;
      if ( v51[16] > 0x10u || *(_BYTE *)(v34 + 16) > 0x10u )
      {
        LOWORD(v68) = 257;
        if ( !v35 )
        {
          v50 = *(_QWORD *)v51;
          if ( *(_BYTE *)(*(_QWORD *)v51 + 8LL) == 16 )
            v50 = **(_QWORD **)(v50 + 16);
          v36 = *(_QWORD *)(v50 + 24);
        }
        v39 = sub_1648A60(72, 2u);
        v40 = v39;
        if ( v39 )
        {
          v41 = (__int64)(v39 - 6);
          v42 = *(_QWORD *)v51;
          if ( *(_BYTE *)(*(_QWORD *)v51 + 8LL) == 16 )
            v42 = **(_QWORD **)(v42 + 16);
          v56 = *(_DWORD *)(v42 + 8) >> 8;
          v43 = (__int64 *)sub_15F9F50(v36, (__int64)&v59, 1);
          v44 = (__int64 *)sub_1646BA0(v43, v56);
          v45 = *(_QWORD *)v51;
          if ( *(_BYTE *)(*(_QWORD *)v51 + 8LL) == 16 || (v45 = *(_QWORD *)v59, *(_BYTE *)(*(_QWORD *)v59 + 8LL) == 16) )
            v44 = sub_16463B0(v44, *(_QWORD *)(v45 + 32));
          sub_15F1EA0((__int64)v40, (__int64)v44, 32, v41, 2, 0);
          v40[7] = v36;
          v40[8] = sub_15F9F50(v36, (__int64)&v59, 1);
          sub_15F9CE0((__int64)v40, (__int64)v51, &v59, 1, (__int64)&v67);
        }
        sub_15FA2E0((__int64)v40, 1);
        v17 = (__int64)sub_1A1C7B0((__int64 *)a1, v40, v66);
      }
      else
      {
        v67.m128i_i8[4] = 0;
        v61 = (__int64 *)v34;
        v17 = sub_15A2E80(v35, (__int64)v51, &v61, 1u, 1u, (__int64)&v67, 0);
      }
      goto LABEL_81;
    }
LABEL_108:
    v37 = v57;
    goto LABEL_109;
  }
LABEL_81:
  if ( a5 != *(_QWORD *)v17 )
  {
    v65 = 1;
    v64 = 3;
    v63.m128i_i64[0] = (__int64)"sroa_cast";
    sub_14EC200(v66, (const __m128i *)&a7, &v63);
    if ( a5 != *(_QWORD *)v17 )
    {
      if ( *(_BYTE *)(v17 + 16) > 0x10u )
      {
        LOWORD(v68) = 257;
        v38 = (_QWORD *)sub_15FDF90(v17, a5, (__int64)&v67, 0);
        v17 = (__int64)sub_1A1C7B0((__int64 *)a1, v38, v66);
      }
      else
      {
        v17 = sub_15A4AD0((__int64 ***)v17, a5);
      }
    }
  }
LABEL_85:
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
  if ( v74 != v73 )
    _libc_free((unsigned __int64)v74);
  return v17;
}
