// Function: sub_31025B0
// Address: 0x31025b0
//
__int64 __fastcall sub_31025B0(
        unsigned __int8 (__fastcall ***a1)(_QWORD, __int64),
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 *v6; // rax
  _QWORD *v8; // rdx
  char v9; // cl
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  unsigned int v15; // r12d
  __int64 *v17; // rax
  __int64 *v18; // r13
  __int64 v19; // rbx
  __int64 *v20; // r15
  __int64 *v21; // rax
  char v22; // al
  unsigned __int64 v23; // rax
  __int64 v24; // r14
  int v25; // ebx
  unsigned int v26; // r15d
  __int64 *v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // r12
  __int64 *v32; // rax
  char v33; // dl
  char v34; // al
  _QWORD *v35; // rax
  _QWORD *v36; // rdx
  _QWORD *v37; // rax
  _QWORD *v38; // rdx
  __int64 v39; // rax
  unsigned __int64 v40; // r10
  unsigned __int64 v41; // rcx
  __int64 v42; // rsi
  _QWORD *v43; // rax
  bool v44; // al
  __int64 v45; // rdx
  unsigned int v46; // r11d
  unsigned int v47; // eax
  _BYTE *v48; // r8
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rax
  _BYTE *v54; // rsi
  _BYTE *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  bool v59; // al
  unsigned __int64 v60; // [rsp+0h] [rbp-360h]
  unsigned __int64 v61; // [rsp+8h] [rbp-358h]
  _BYTE *v62; // [rsp+10h] [rbp-350h]
  unsigned __int64 v63; // [rsp+18h] [rbp-348h]
  __int64 v64; // [rsp+18h] [rbp-348h]
  unsigned __int64 v65; // [rsp+20h] [rbp-340h]
  unsigned int v66; // [rsp+20h] [rbp-340h]
  bool v67; // [rsp+20h] [rbp-340h]
  unsigned __int64 v68; // [rsp+28h] [rbp-338h]
  __int64 *v69; // [rsp+38h] [rbp-328h]
  __int64 v74; // [rsp+70h] [rbp-2F0h] BYREF
  _BYTE *v75; // [rsp+78h] [rbp-2E8h]
  __int64 v76; // [rsp+80h] [rbp-2E0h]
  int v77; // [rsp+88h] [rbp-2D8h]
  char v78; // [rsp+8Ch] [rbp-2D4h]
  _BYTE v79[32]; // [rsp+90h] [rbp-2D0h] BYREF
  __int64 v80; // [rsp+B0h] [rbp-2B0h] BYREF
  __int64 *v81; // [rsp+B8h] [rbp-2A8h]
  __int64 v82; // [rsp+C0h] [rbp-2A0h]
  int v83; // [rsp+C8h] [rbp-298h]
  char v84; // [rsp+CCh] [rbp-294h]
  _BYTE v85[32]; // [rsp+D0h] [rbp-290h] BYREF
  __m128i v86; // [rsp+F0h] [rbp-270h] BYREF
  __int64 v87; // [rsp+100h] [rbp-260h]
  __int64 v88; // [rsp+108h] [rbp-258h]
  __int64 v89; // [rsp+110h] [rbp-250h]
  unsigned __int64 v90; // [rsp+118h] [rbp-248h]
  __int64 v91; // [rsp+120h] [rbp-240h]
  __int64 v92; // [rsp+128h] [rbp-238h]
  __int16 v93; // [rsp+130h] [rbp-230h]
  _QWORD v94[7]; // [rsp+140h] [rbp-220h] BYREF
  char v95; // [rsp+178h] [rbp-1E8h] BYREF
  char *v96; // [rsp+180h] [rbp-1E0h]
  __int64 v97; // [rsp+188h] [rbp-1D8h]
  char v98; // [rsp+190h] [rbp-1D0h] BYREF
  char *v99; // [rsp+1C0h] [rbp-1A0h]
  __int64 v100; // [rsp+1C8h] [rbp-198h]
  char v101; // [rsp+1D0h] [rbp-190h] BYREF
  char *v102; // [rsp+1F0h] [rbp-170h]
  __int64 v103; // [rsp+1F8h] [rbp-168h]
  char v104; // [rsp+200h] [rbp-160h] BYREF
  char *v105; // [rsp+250h] [rbp-110h]
  __int64 v106; // [rsp+258h] [rbp-108h]
  char v107; // [rsp+260h] [rbp-100h] BYREF
  char *v108; // [rsp+300h] [rbp-60h]
  __int64 v109; // [rsp+308h] [rbp-58h]
  char v110; // [rsp+310h] [rbp-50h] BYREF
  __int16 v111; // [rsp+320h] [rbp-40h]
  __int64 v112; // [rsp+328h] [rbp-38h]

  v6 = (__int64 *)v79;
  v8 = *(_QWORD **)(a2 + 32);
  v74 = 0;
  v75 = v79;
  v76 = 4;
  v77 = 0;
  v78 = 1;
  if ( a3 == *v8 )
  {
    v9 = 1;
    v10 = *(_QWORD *)(a3 + 16);
    if ( !v10 )
    {
      v80 = 0;
      v81 = (__int64 *)v85;
      v82 = 4;
      v83 = 0;
      v84 = 1;
      goto LABEL_16;
    }
LABEL_3:
    while ( 1 )
    {
      v11 = *(_QWORD *)(v10 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v11 - 30) <= 0xAu )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        goto LABEL_15;
    }
    while ( 1 )
    {
      v12 = *(_QWORD *)(v11 + 40);
      if ( v9 )
      {
        v13 = v75;
        v14 = &v75[8 * HIDWORD(v76)];
        if ( v75 != (_BYTE *)v14 )
        {
          while ( v12 != *v13 )
          {
            if ( v14 == ++v13 )
              goto LABEL_12;
          }
LABEL_9:
          v15 = 0;
          if ( !v9 )
            goto LABEL_22;
          return v15;
        }
      }
      else
      {
        v17 = sub_C8CA60((__int64)&v74, v12);
        v9 = v78;
        if ( v17 )
          goto LABEL_9;
      }
LABEL_12:
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        break;
      while ( 1 )
      {
        v11 = *(_QWORD *)(v10 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v11 - 30) <= 0xAu )
          break;
        v10 = *(_QWORD *)(v10 + 8);
        if ( !v10 )
          goto LABEL_15;
      }
    }
  }
  else
  {
    sub_3102230(a2, a3, (__int64)&v74, (__int64)&v74, a5, a6);
    v9 = v78;
    v10 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 16LL);
    if ( v10 )
      goto LABEL_3;
  }
LABEL_15:
  v80 = 0;
  v81 = (__int64 *)v85;
  v6 = (__int64 *)v75;
  v82 = 4;
  v83 = 0;
  v84 = 1;
  if ( v9 )
  {
LABEL_16:
    v18 = &v6[HIDWORD(v76)];
    goto LABEL_17;
  }
  v18 = (__int64 *)&v75[8 * (unsigned int)v76];
LABEL_17:
  if ( v6 == v18 )
    goto LABEL_20;
  while ( 1 )
  {
    v19 = *v6;
    v20 = v6;
    if ( (unsigned __int64)*v6 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v18 == ++v6 )
      goto LABEL_20;
  }
  if ( v18 == v6 )
  {
LABEL_20:
    v15 = 1;
    goto LABEL_21;
  }
  while ( 1 )
  {
    if ( (**a1)(a1, v19) )
      goto LABEL_64;
    if ( !(unsigned __int8)sub_B19720(a4, a3, v19) )
    {
      v23 = *(_QWORD *)(v19 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v23 != v19 + 48 )
      {
        if ( !v23 )
          BUG();
        v24 = v23 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v23 - 24) - 30 <= 0xA )
        {
          v25 = sub_B46E30(v24);
          if ( v25 )
            break;
        }
      }
    }
LABEL_28:
    v21 = v20 + 1;
    if ( v20 + 1 != v18 )
    {
      while ( 1 )
      {
        v19 = *v21;
        v20 = v21;
        if ( (unsigned __int64)*v21 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v18 == ++v21 )
          goto LABEL_31;
      }
      if ( v18 != v21 )
        continue;
    }
LABEL_31:
    v22 = v84;
    v15 = 1;
    goto LABEL_32;
  }
  v69 = v20;
  v26 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v31 = sub_B46EC0(v24, v26);
      if ( !v84 )
        goto LABEL_48;
      v32 = v81;
      v28 = HIDWORD(v82);
      v27 = &v81[HIDWORD(v82)];
      if ( v81 != v27 )
      {
        while ( v31 != *v32 )
        {
          if ( v27 == ++v32 )
            goto LABEL_56;
        }
        goto LABEL_46;
      }
LABEL_56:
      if ( HIDWORD(v82) < (unsigned int)v82 )
      {
        v34 = a3 != v31;
        ++HIDWORD(v82);
        *v27 = v31;
        ++v80;
      }
      else
      {
LABEL_48:
        sub_C8CC70((__int64)&v80, v31, (__int64)v27, v28, v29, v30);
        v34 = v33 & (a3 != v31);
      }
      if ( v34 )
        break;
LABEL_46:
      if ( v25 == ++v26 )
        goto LABEL_47;
    }
    if ( !v78 )
    {
      if ( !sub_C8CA60((__int64)&v74, v31) )
        goto LABEL_59;
      goto LABEL_46;
    }
    v35 = v75;
    v36 = &v75[8 * HIDWORD(v76)];
    if ( v75 != (_BYTE *)v36 )
    {
      while ( v31 != *v35 )
      {
        if ( v36 == ++v35 )
          goto LABEL_59;
      }
      goto LABEL_46;
    }
LABEL_59:
    if ( *(_BYTE *)(a2 + 84) )
      break;
    if ( sub_C8CA60(a2 + 56, v31) )
      goto LABEL_64;
LABEL_66:
    v39 = sub_AA54C0(v31);
    if ( !v39 )
      goto LABEL_64;
    v40 = *(_QWORD *)(v39 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v40 == v39 + 48 || !v40 || (v41 = v40 - 24, (unsigned int)*(unsigned __int8 *)(v40 - 24) - 30 > 0xA) )
      BUG();
    if ( *(_BYTE *)(v40 - 24) != 31 || (*(_DWORD *)(v40 - 20) & 0x7FFFFFF) != 3 )
      goto LABEL_64;
    v42 = *(_QWORD *)(v40 - 120);
    if ( *(_BYTE *)v42 == 17 )
    {
      v43 = *(_QWORD **)(v42 + 24);
      if ( *(_DWORD *)(v42 + 32) > 0x40u )
        v43 = (_QWORD *)*v43;
      v44 = *(_QWORD *)(v40 - 32LL * (v43 != 0) - 56) == v31;
    }
    else
    {
      if ( (unsigned __int8)(*(_BYTE *)v42 - 82) > 1u )
        goto LABEL_64;
      v45 = *(_QWORD *)(v42 - 64);
      v46 = *(_WORD *)(v42 + 2) & 0x3F;
      if ( *(_BYTE *)v45 == 84 && *(_QWORD *)(v45 + 40) == **(_QWORD **)(a2 + 32) )
      {
        v48 = *(_BYTE **)(v42 - 32);
      }
      else
      {
        v63 = v40 - 24;
        v65 = *(_QWORD *)(v39 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        v47 = sub_B52F50(v46);
        v40 = v65;
        v41 = v63;
        v46 = v47;
        v45 = *(_QWORD *)(v42 - 32);
        if ( *(_BYTE *)v45 != 84 || *(_QWORD *)(v45 + 40) != **(_QWORD **)(a2 + 32) )
          goto LABEL_64;
        v48 = *(_BYTE **)(v42 - 64);
      }
      v60 = v41;
      v61 = v40;
      v62 = v48;
      v66 = v46;
      v64 = v45;
      v49 = sub_AA4B30(v31);
      v94[4] = &v95;
      v96 = &v98;
      v97 = 0x600000000LL;
      v99 = &v101;
      v100 = 0x400000000LL;
      v102 = &v104;
      v103 = 0xA00000000LL;
      v105 = &v107;
      v106 = 0x800000000LL;
      v108 = &v110;
      v111 = 768;
      memset(v94, 0, 32);
      v94[5] = 0;
      v94[6] = 8;
      v109 = 0;
      v110 = 0;
      v112 = 0;
      sub_AE1EA0((__int64)v94, v49 + 312);
      v50 = sub_D4B130(a2);
      v51 = *(_QWORD *)(v64 - 8);
      if ( (*(_DWORD *)(v64 + 4) & 0x7FFFFFF) != 0 )
      {
        v52 = 0;
        while ( v50 != *(_QWORD *)(v51 + 32LL * *(unsigned int *)(v64 + 72) + 8 * v52) )
        {
          if ( (*(_DWORD *)(v64 + 4) & 0x7FFFFFF) == (_DWORD)++v52 )
            goto LABEL_100;
        }
        v53 = 32 * v52;
      }
      else
      {
LABEL_100:
        v53 = 0x1FFFFFFFE0LL;
      }
      v54 = *(_BYTE **)(v51 + v53);
      v90 = v60;
      v86.m128i_i64[0] = (__int64)v94;
      v86.m128i_i64[1] = 0;
      v88 = a4;
      v93 = 257;
      v87 = 0;
      v89 = 0;
      v68 = v66 | v68 & 0xFFFFFF0000000000LL;
      v91 = 0;
      v92 = 0;
      v55 = (_BYTE *)sub_10197D0(v68, v54, v62, &v86);
      if ( !v55 || *v55 > 0x15u )
      {
        sub_AE4030(v94, (__int64)v54);
        goto LABEL_64;
      }
      if ( v31 == *(_QWORD *)(v61 - 56) )
        v59 = sub_AD7890((__int64)v55, (__int64)v54, v56, v57, v58);
      else
        v59 = sub_AD7930(v55, (__int64)v54, v56, v57, v58);
      v67 = v59;
      sub_AE4030(v94, (__int64)v54);
      v44 = v67;
    }
    if ( !v44 )
      goto LABEL_64;
    if ( v25 == ++v26 )
    {
LABEL_47:
      v20 = v69;
      goto LABEL_28;
    }
  }
  v37 = *(_QWORD **)(a2 + 64);
  v38 = &v37[*(unsigned int *)(a2 + 76)];
  if ( v37 == v38 )
    goto LABEL_66;
  while ( v31 != *v37 )
  {
    if ( v38 == ++v37 )
      goto LABEL_66;
  }
LABEL_64:
  v22 = v84;
  v15 = 0;
LABEL_32:
  if ( !v22 )
    _libc_free((unsigned __int64)v81);
LABEL_21:
  if ( !v78 )
LABEL_22:
    _libc_free((unsigned __int64)v75);
  return v15;
}
