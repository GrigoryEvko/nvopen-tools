// Function: sub_191D810
// Address: 0x191d810
//
__int64 __fastcall sub_191D810(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rcx
  _QWORD *v15; // rdx
  __int64 v17; // rsi
  __int64 v18; // r8
  __int64 v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  unsigned __int64 v22; // r15
  char v23; // al
  __int64 *v24; // rsi
  unsigned __int8 v25; // al
  char v27; // dl
  unsigned int v28; // ecx
  __int64 *v29; // rax
  __int64 v30; // rax
  char *v31; // rax
  unsigned int v32; // ecx
  __int64 v33; // rbx
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // rdx
  _QWORD *v37; // r8
  int v38; // esi
  unsigned int v39; // ecx
  _QWORD *v40; // rax
  __int64 v41; // r9
  int v42; // edx
  int v43; // r8d
  int v44; // r9d
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rbx
  __int64 v48; // rdx
  __int64 v49; // r12
  __int64 *v50; // rax
  __int64 v51; // rax
  int v52; // r15d
  __int64 *v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned int v56; // ebx
  unsigned int v57; // r8d
  unsigned __int8 v58; // cl
  __int64 v59; // rcx
  int v60; // eax
  unsigned int v61; // esi
  unsigned __int32 v62; // ecx
  unsigned __int32 v63; // r8d
  unsigned int v64; // r9d
  unsigned int v65; // eax
  __int64 v66; // r14
  __int8 v67; // di
  __int64 v68; // rbx
  __int64 v69; // r9
  __int64 *v70; // rax
  char v71; // r11
  _QWORD *v72; // r8
  int v73; // ecx
  __int64 v74; // rsi
  unsigned int v75; // edx
  _QWORD *v76; // rax
  __int64 v77; // r10
  int v78; // eax
  __int64 v79; // rax
  int v80; // eax
  int v81; // r11d
  _QWORD *v82; // r10
  _DWORD *v83; // rax
  double v84; // xmm4_8
  double v85; // xmm5_8
  char v86; // al
  int v87; // r8d
  int v88; // r9d
  __int64 v89; // rax
  int v90; // r10d
  _QWORD *v91; // r11
  int v92; // [rsp+0h] [rbp-180h]
  __int64 v93; // [rsp+8h] [rbp-178h]
  __int64 v94; // [rsp+10h] [rbp-170h]
  __int64 v95; // [rsp+10h] [rbp-170h]
  __int64 v96; // [rsp+18h] [rbp-168h]
  unsigned int v97; // [rsp+18h] [rbp-168h]
  __int64 v98; // [rsp+28h] [rbp-158h] BYREF
  __int64 v99; // [rsp+30h] [rbp-150h] BYREF
  __int64 v100; // [rsp+38h] [rbp-148h]
  __m128i v101; // [rsp+40h] [rbp-140h] BYREF
  _QWORD *v102; // [rsp+50h] [rbp-130h] BYREF
  __int64 v103; // [rsp+58h] [rbp-128h]
  __int64 v104; // [rsp+60h] [rbp-120h]
  char v105; // [rsp+150h] [rbp-30h] BYREF

  if ( *(_BYTE *)(a2 + 16) == 78 )
  {
    v46 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v46 + 16) && (*(_BYTE *)(v46 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v46 + 36) - 35) <= 3 )
      goto LABEL_39;
  }
  v12 = sub_15F2050(a2);
  v13 = sub_1632FA0(v12);
  v14 = *(_QWORD *)(a1 + 32);
  v15 = *(_QWORD **)(a1 + 24);
  v17 = *(_QWORD *)(a1 + 40);
  v101.m128i_i64[0] = v13;
  v101.m128i_i64[1] = v14;
  v102 = v15;
  v103 = v17;
  v104 = 0;
  v19 = sub_13E3350(a2, &v101, 0, 1, v18);
  v22 = v19;
  if ( v19 )
  {
    if ( *(_QWORD *)(a2 + 8) )
    {
      sub_164D160(a2, v19, a3, a4, a5, a6, v20, v21, a9, a10);
      if ( !(unsigned __int8)sub_1AE9990(a2, *(_QWORD *)(a1 + 32)) )
        goto LABEL_5;
      goto LABEL_33;
    }
    if ( (unsigned __int8)sub_1AE9990(a2, *(_QWORD *)(a1 + 32)) )
    {
LABEL_33:
      sub_190ACD0(a1 + 152, a2);
      v45 = *(unsigned int *)(a1 + 680);
      if ( (unsigned int)v45 >= *(_DWORD *)(a1 + 684) )
      {
        sub_16CD150(a1 + 672, (const void *)(a1 + 688), 0, 8, v43, v44);
        v45 = *(unsigned int *)(a1 + 680);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v45) = a2;
      ++*(_DWORD *)(a1 + 680);
LABEL_5:
      if ( *(_QWORD *)a1 )
      {
        v23 = *(_BYTE *)(*(_QWORD *)v22 + 8LL);
        if ( v23 == 16 )
          v23 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v22 + 16LL) + 8LL);
        if ( v23 == 15 )
        {
          v24 = (__int64 *)v22;
          LODWORD(v22) = 1;
          sub_14134C0(*(_QWORD *)a1, v24);
          return (unsigned int)v22;
        }
      }
      goto LABEL_15;
    }
  }
  v25 = *(_BYTE *)(a2 + 16);
  if ( v25 <= 0x17u )
    goto LABEL_17;
  if ( v25 == 78 )
  {
    v55 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v55 + 16) && (*(_BYTE *)(v55 + 33) & 0x20) != 0 && *(_DWORD *)(v55 + 36) == 4 )
      return sub_1918A30(a1, a2);
    goto LABEL_47;
  }
  if ( v25 == 54 )
  {
    LODWORD(v22) = sub_191D600(a1, a2, a3, a4, a5, a6, v20, v21, a9, a10);
    if ( !(_BYTE)v22 )
    {
      v60 = sub_1911FD0(a1 + 152, a2);
      sub_1910810(a1, v60, a2, *(_QWORD *)(a2 + 40));
      return (unsigned int)v22;
    }
LABEL_15:
    LODWORD(v22) = 1;
    return (unsigned int)v22;
  }
  if ( v25 != 26 )
  {
LABEL_17:
    if ( v25 == 27 )
    {
      v27 = *(_BYTE *)(a2 + 23);
      v28 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      if ( (v27 & 0x40) != 0 )
        v29 = *(__int64 **)(a2 - 8);
      else
        v29 = (__int64 *)(a2 - 24LL * v28);
      v30 = *v29;
      v101.m128i_i64[0] = 0;
      v101.m128i_i64[1] = 1;
      v93 = v30;
      v94 = *(_QWORD *)(a2 + 40);
      v31 = (char *)&v102;
      do
      {
        *(_QWORD *)v31 = -8;
        v31 += 16;
      }
      while ( v31 != &v105 );
      v32 = v28 >> 1;
      if ( v32 )
      {
        v33 = 24;
        v34 = 48LL * (v32 - 1) + 72;
        while ( 1 )
        {
          if ( (v27 & 0x40) != 0 )
            v35 = *(_QWORD *)(a2 - 8);
          else
            v35 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          v36 = *(_QWORD *)(v35 + v33);
          v98 = v36;
          if ( (v101.m128i_i8[8] & 1) != 0 )
          {
            v37 = &v102;
            v38 = 15;
          }
          else
          {
            v61 = v103;
            v37 = v102;
            if ( !(_DWORD)v103 )
            {
              v62 = v101.m128i_u32[2];
              ++v101.m128i_i64[0];
              v40 = 0;
              v63 = ((unsigned __int32)v101.m128i_i32[2] >> 1) + 1;
LABEL_57:
              v64 = 3 * v61;
              goto LABEL_58;
            }
            v38 = v103 - 1;
          }
          v39 = v38 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
          v40 = &v37[2 * v39];
          v41 = *v40;
          if ( v36 == *v40 )
          {
            v42 = *((_DWORD *)v40 + 2) + 1;
            goto LABEL_29;
          }
          v81 = 1;
          v82 = 0;
          while ( v41 != -8 )
          {
            if ( v41 != -16 || v82 )
              v40 = v82;
            v90 = v81 + 1;
            v39 = v38 & (v81 + v39);
            v91 = &v37[2 * v39];
            v41 = *v91;
            if ( v36 == *v91 )
            {
              v40 = &v37[2 * v39];
              v42 = *((_DWORD *)v91 + 2) + 1;
              goto LABEL_29;
            }
            v81 = v90;
            v82 = v40;
            v40 = &v37[2 * v39];
          }
          v62 = v101.m128i_u32[2];
          v64 = 48;
          v61 = 16;
          if ( v82 )
            v40 = v82;
          ++v101.m128i_i64[0];
          v63 = ((unsigned __int32)v101.m128i_i32[2] >> 1) + 1;
          if ( (v101.m128i_i8[8] & 1) == 0 )
          {
            v61 = v103;
            goto LABEL_57;
          }
LABEL_58:
          if ( 4 * v63 >= v64 )
          {
            v61 *= 2;
LABEL_83:
            sub_1917CA0((__int64)&v101, v61);
            sub_190F380((__int64)&v101, &v98, &v99);
            v40 = (_QWORD *)v99;
            v36 = v98;
            v62 = v101.m128i_u32[2];
            goto LABEL_60;
          }
          if ( v61 - v101.m128i_i32[3] - v63 <= v61 >> 3 )
            goto LABEL_83;
LABEL_60:
          v101.m128i_i32[2] = (2 * (v62 >> 1) + 2) | v62 & 1;
          if ( *v40 != -8 )
            --v101.m128i_i32[3];
          *v40 = v36;
          v42 = 1;
          *((_DWORD *)v40 + 2) = 0;
LABEL_29:
          v33 += 48;
          *((_DWORD *)v40 + 2) = v42;
          if ( v33 == v34 )
          {
            v65 = (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1;
            v66 = v65 - 1;
            if ( v65 != 1 )
            {
              v27 = *(_BYTE *)(a2 + 23);
              goto LABEL_65;
            }
            LODWORD(v22) = 0;
            v71 = v101.m128i_i8[8] & 1;
LABEL_80:
            if ( !v71 )
              j___libc_free_0(v102);
            return (unsigned int)v22;
          }
          v27 = *(_BYTE *)(a2 + 23);
        }
      }
      v66 = 0xFFFFFFFFLL;
LABEL_65:
      v67 = v101.m128i_i8[8];
      v68 = 0;
      LODWORD(v22) = 0;
      while ( 1 )
      {
        v79 = 24;
        if ( (_DWORD)v68 != -2 )
          v79 = 24LL * (unsigned int)(2 * v68 + 3);
        if ( (v27 & 0x40) != 0 )
        {
          v69 = *(_QWORD *)(a2 - 8);
          ++v68;
          v70 = (__int64 *)(v69 + v79);
          v71 = v67 & 1;
          if ( (v67 & 1) != 0 )
            goto LABEL_67;
        }
        else
        {
          ++v68;
          v69 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          v70 = (__int64 *)(v69 + v79);
          v71 = v67 & 1;
          if ( (v67 & 1) != 0 )
          {
LABEL_67:
            v72 = &v102;
            v73 = 15;
            goto LABEL_68;
          }
        }
        v72 = v102;
        if ( !(_DWORD)v103 )
          goto LABEL_71;
        v73 = v103 - 1;
LABEL_68:
        v74 = *v70;
        v75 = v73 & (((unsigned int)*v70 >> 9) ^ ((unsigned int)*v70 >> 4));
        v76 = &v72[2 * v75];
        v77 = *v76;
        if ( v74 == *v76 )
        {
LABEL_69:
          if ( *((_DWORD *)v76 + 2) == 1 )
          {
            v100 = v74;
            v99 = v94;
            v78 = sub_19166D0(a1, v93, *(_QWORD *)(v69 + 24LL * (unsigned int)(2 * v68)), &v99, 1);
            v67 = v101.m128i_i8[8];
            LODWORD(v22) = v78 | v22;
            v71 = v101.m128i_i8[8] & 1;
          }
        }
        else
        {
          v80 = 1;
          while ( v77 != -8 )
          {
            v75 = v73 & (v80 + v75);
            v92 = v80 + 1;
            v76 = &v72[2 * v75];
            v77 = *v76;
            if ( v74 == *v76 )
              goto LABEL_69;
            v80 = v92;
          }
        }
LABEL_71:
        if ( v66 == v68 )
          goto LABEL_80;
        v27 = *(_BYTE *)(a2 + 23);
      }
    }
LABEL_47:
    if ( !*(_BYTE *)(*(_QWORD *)a2 + 8LL) )
      goto LABEL_39;
    v56 = *(_DWORD *)(a1 + 360);
    v57 = sub_1911FD0(a1 + 152, a2);
    v58 = *(_BYTE *)(a2 + 16) - 25;
    if ( v58 > 0x34u )
    {
      v59 = *(_QWORD *)(a2 + 40);
    }
    else
    {
      v22 = 0x100000100003FFuLL >> v58;
      v59 = *(_QWORD *)(a2 + 40);
      LODWORD(v22) = (v22 & 1) == 0;
      if ( !(_DWORD)v22 )
        goto LABEL_50;
    }
    if ( v57 >= v56 )
    {
      sub_1910810(a1, v57, a2, v59);
      goto LABEL_39;
    }
    v97 = v57;
    v83 = sub_1910330(a1, v59, v57);
    v57 = v97;
    v22 = (unsigned __int64)v83;
    if ( v83 )
    {
      if ( v83 != (_DWORD *)a2 )
      {
        sub_1909530(a2, (__int64)v83);
        sub_164D160(a2, v22, a3, a4, a5, a6, v84, v85, a9, a10);
        if ( *(_QWORD *)a1 )
        {
          v86 = *(_BYTE *)(*(_QWORD *)v22 + 8LL);
          if ( v86 == 16 )
            v86 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v22 + 16LL) + 8LL);
          if ( v86 == 15 )
            sub_14134C0(*(_QWORD *)a1, (__int64 *)v22);
        }
        sub_190ACD0(a1 + 152, a2);
        v89 = *(unsigned int *)(a1 + 680);
        if ( (unsigned int)v89 >= *(_DWORD *)(a1 + 684) )
        {
          sub_16CD150(a1 + 672, (const void *)(a1 + 688), 0, 8, v87, v88);
          v89 = *(unsigned int *)(a1 + 680);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v89) = a2;
        ++*(_DWORD *)(a1 + 680);
        goto LABEL_15;
      }
LABEL_39:
      LODWORD(v22) = 0;
      return (unsigned int)v22;
    }
    v59 = *(_QWORD *)(a2 + 40);
LABEL_50:
    sub_1910810(a1, v57, a2, v59);
    return (unsigned int)v22;
  }
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 3 )
    goto LABEL_39;
  v47 = *(_QWORD *)(a2 - 72);
  if ( *(_BYTE *)(v47 + 16) > 0x10u )
  {
    v48 = *(_QWORD *)(a2 - 24);
    LODWORD(v22) = 0;
    v96 = *(_QWORD *)(a2 - 48);
    if ( v96 != v48 )
    {
      v95 = *(_QWORD *)(a2 - 24);
      v49 = *(_QWORD *)(a2 + 40);
      v50 = (__int64 *)sub_157E9C0(v48);
      v51 = sub_159C4F0(v50);
      v99 = v49;
      v100 = v95;
      v52 = sub_19166D0(a1, v47, v51, &v99, 1);
      v53 = (__int64 *)sub_157E9C0(v96);
      v54 = sub_159C540(v53);
      v101.m128i_i64[0] = v49;
      v101.m128i_i64[1] = v96;
      LODWORD(v22) = sub_19166D0(a1, v47, v54, &v101, 1) | v52;
    }
    return (unsigned int)v22;
  }
  return sub_191A210(a1, a2);
}
