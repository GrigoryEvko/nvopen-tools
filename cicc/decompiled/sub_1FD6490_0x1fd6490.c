// Function: sub_1FD6490
// Address: 0x1fd6490
//
__int64 __fastcall sub_1FD6490(_QWORD *a1, __int64 a2)
{
  __int64 **v2; // r15
  __int64 v4; // rsi
  __int64 **v5; // rdx
  __int64 v6; // rdi
  int v7; // r8d
  int v8; // r9d
  _BYTE *v9; // r12
  __int64 v10; // rcx
  char v11; // al
  __int64 v12; // rbx
  __int64 v13; // r14
  int v14; // r9d
  __int64 v15; // rdi
  __int64 (*v16)(); // rax
  unsigned __int8 v17; // bl
  __int64 v18; // r10
  __int64 v19; // r14
  __int64 **v20; // rax
  __int64 v21; // r13
  int v22; // r13d
  int v23; // eax
  __int64 v24; // rdx
  int v25; // r8d
  int v26; // r14d
  __int64 v27; // r13
  char v28; // cl
  __m128i *v29; // rdx
  __int64 *v30; // rax
  __int64 *v31; // r8
  __int64 **v32; // r13
  __int64 *v33; // r14
  int v34; // r15d
  unsigned __int64 v35; // r15
  unsigned __int64 v36; // r15
  unsigned int v37; // edx
  int v38; // r8d
  int v39; // r9d
  char v40; // al
  __int64 v41; // rdx
  unsigned __int64 v42; // rax
  __int64 v43; // rdx
  char v44; // al
  __int64 v45; // r11
  __int64 v46; // rsi
  bool v47; // dl
  __int64 v48; // rdi
  __int64 (*v49)(); // r10
  int v50; // ecx
  __int64 v51; // rax
  unsigned __int64 v52; // rax
  unsigned int v53; // edx
  unsigned int v54; // eax
  char v55; // al
  unsigned __int64 v56; // r15
  __int64 v57; // r10
  __int64 v58; // r14
  char v59; // di
  unsigned int v60; // eax
  int v61; // eax
  __int64 v62; // rax
  int v63; // eax
  __int64 (*v64)(); // rax
  unsigned int v65; // r12d
  __int64 v67; // rax
  int v68; // ecx
  __int64 *v69; // rax
  __int64 v70; // [rsp+0h] [rbp-250h]
  __int64 v71; // [rsp+8h] [rbp-248h]
  int v72; // [rsp+10h] [rbp-240h]
  __int64 v73; // [rsp+30h] [rbp-220h]
  __int64 v74; // [rsp+38h] [rbp-218h]
  __int64 v75; // [rsp+48h] [rbp-208h]
  __int64 v76; // [rsp+50h] [rbp-200h]
  __int64 v77; // [rsp+58h] [rbp-1F8h]
  __int64 v78; // [rsp+58h] [rbp-1F8h]
  __int64 *v79; // [rsp+60h] [rbp-1F0h]
  unsigned __int8 v80; // [rsp+68h] [rbp-1E8h]
  unsigned __int8 v81; // [rsp+68h] [rbp-1E8h]
  __int64 v82; // [rsp+68h] [rbp-1E8h]
  __int64 v83; // [rsp+68h] [rbp-1E8h]
  __int64 v84; // [rsp+68h] [rbp-1E8h]
  __int64 v85; // [rsp+70h] [rbp-1E0h]
  int v86; // [rsp+70h] [rbp-1E0h]
  __int64 v87; // [rsp+78h] [rbp-1D8h]
  unsigned __int64 v88; // [rsp+78h] [rbp-1D8h]
  __int64 v89; // [rsp+78h] [rbp-1D8h]
  __int64 v90; // [rsp+78h] [rbp-1D8h]
  int v91; // [rsp+78h] [rbp-1D8h]
  __int64 v93; // [rsp+88h] [rbp-1C8h]
  unsigned __int64 v94; // [rsp+88h] [rbp-1C8h]
  unsigned __int64 v95; // [rsp+88h] [rbp-1C8h]
  unsigned __int64 v96; // [rsp+88h] [rbp-1C8h]
  unsigned __int8 v97; // [rsp+9Bh] [rbp-1B5h] BYREF
  unsigned int v98; // [rsp+9Ch] [rbp-1B4h] BYREF
  __int64 v99; // [rsp+A0h] [rbp-1B0h] BYREF
  __int64 v100; // [rsp+A8h] [rbp-1A8h]
  __int64 v101; // [rsp+B0h] [rbp-1A0h] BYREF
  __int64 v102; // [rsp+B8h] [rbp-198h]
  __int64 v103; // [rsp+C0h] [rbp-190h] BYREF
  __int64 v104; // [rsp+C8h] [rbp-188h]
  __m128i v105; // [rsp+D0h] [rbp-180h] BYREF
  __m128i v106; // [rsp+E0h] [rbp-170h] BYREF
  __m128i v107; // [rsp+F0h] [rbp-160h] BYREF
  _BYTE *v108; // [rsp+100h] [rbp-150h] BYREF
  __int64 v109; // [rsp+108h] [rbp-148h]
  _BYTE v110[64]; // [rsp+110h] [rbp-140h] BYREF
  unsigned __int64 v111[2]; // [rsp+150h] [rbp-100h] BYREF
  _BYTE v112[240]; // [rsp+160h] [rbp-F0h] BYREF

  v2 = (__int64 **)a2;
  *(_DWORD *)(a2 + 464) = 0;
  *(_DWORD *)(a2 + 672) = 0;
  v4 = a1[12];
  v5 = (__int64 **)*v2;
  v6 = a1[14];
  v108 = v110;
  v109 = 0x400000000LL;
  sub_20C7CE0(v6, v4, v5, &v108, 0, 0);
  v9 = (_BYTE *)a1[14];
  v10 = 0;
  v111[0] = (unsigned __int64)v112;
  v105.m128i_i64[1] = 0x200000000LL;
  v11 = *((_BYTE *)v2 + 8);
  v111[1] = 0x400000000LL;
  v12 = a1[12];
  v105.m128i_i64[0] = (__int64)&v106;
  if ( (v11 & 1) != 0 )
  {
    v106.m128i_i32[0] = 40;
    v10 = 1;
    v105.m128i_i32[2] = 1;
  }
  if ( (v11 & 2) != 0 )
  {
    v106.m128i_i32[v10] = 58;
    v10 = (unsigned int)++v105.m128i_i32[2];
    if ( (v11 & 8) == 0 )
      goto LABEL_5;
    if ( (unsigned int)v10 >= v105.m128i_i32[3] )
    {
      sub_16CD150((__int64)&v105, &v106, 0, 4, v7, v8);
      v10 = v105.m128i_u32[2];
    }
LABEL_73:
    *(_DWORD *)(v105.m128i_i64[0] + 4 * v10) = 12;
    v10 = (unsigned int)++v105.m128i_i32[2];
    goto LABEL_5;
  }
  if ( (v11 & 8) != 0 )
    goto LABEL_73;
LABEL_5:
  v13 = sub_1560040((__int64 *)**v2, 0, (unsigned int *)v105.m128i_i64[0], v10);
  if ( (__m128i *)v105.m128i_i64[0] != &v106 )
    _libc_free(v105.m128i_u64[0]);
  sub_1F431C0(*((_DWORD *)v2 + 4), *v2, v13, (__int64)v111, v9, v12);
  v15 = a1[14];
  v16 = *(__int64 (**)())(*(_QWORD *)v15 + 1216LL);
  if ( v16 != sub_1FD3420
    && !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, bool, unsigned __int64 *, __int64))v16)(
          v15,
          *((unsigned int *)v2 + 4),
          *(_QWORD *)(a1[5] + 8LL),
          ((_BYTE)v2[1] & 4) != 0,
          v111,
          **v2) )
  {
    goto LABEL_63;
  }
  if ( (_DWORD)v109 )
  {
    v93 = 0;
    v77 = 16LL * (unsigned int)v109;
    do
    {
      v17 = v108[v93];
      v18 = *(_QWORD *)&v108[v93 + 8];
      LOBYTE(v99) = v17;
      v19 = a1[14];
      v20 = (__int64 **)*v2;
      v100 = v18;
      v21 = (__int64)*v20;
      if ( v17 )
      {
        v22 = *(unsigned __int8 *)(v19 + v17 + 1155);
        v23 = *(unsigned __int8 *)(v19 + v17 + 1040);
      }
      else
      {
        v89 = v18;
        if ( sub_1F58D20((__int64)&v99) )
        {
          v105.m128i_i8[0] = 0;
          v105.m128i_i64[1] = 0;
          LOBYTE(v101) = 0;
          sub_1F426C0(v19, v21, (unsigned int)v99, v89, (__int64)&v105, (unsigned int *)&v103, &v101);
          v22 = (unsigned __int8)v101;
          v57 = v89;
          v82 = a1[14];
          v75 = **v2;
        }
        else
        {
          sub_1F40D10((__int64)&v105, v19, v21, v99, v100);
          v57 = v89;
          LOBYTE(v101) = v105.m128i_i8[8];
          v102 = v106.m128i_i64[0];
          if ( v105.m128i_i8[8] )
          {
            v22 = *(unsigned __int8 *)(v19 + v105.m128i_u8[8] + 1155);
          }
          else
          {
            v83 = v106.m128i_i64[0];
            if ( sub_1F58D20((__int64)&v101) )
            {
              v105.m128i_i8[0] = 0;
              v105.m128i_i64[1] = 0;
              LOBYTE(v98) = 0;
              sub_1F426C0(v19, v21, (unsigned int)v101, v83, (__int64)&v105, (unsigned int *)&v103, &v98);
              v22 = (unsigned __int8)v98;
              v57 = v89;
            }
            else
            {
              sub_1F40D10((__int64)&v105, v19, v21, v101, v102);
              v57 = v89;
              LOBYTE(v103) = v105.m128i_i8[8];
              v104 = v106.m128i_i64[0];
              if ( v105.m128i_i8[8] )
              {
                v22 = *(unsigned __int8 *)(v19 + v105.m128i_u8[8] + 1155);
              }
              else
              {
                v84 = v106.m128i_i64[0];
                if ( sub_1F58D20((__int64)&v103) )
                {
                  v105.m128i_i8[0] = 0;
                  v105.m128i_i64[1] = 0;
                  v97 = 0;
                  sub_1F426C0(v19, v21, (unsigned int)v103, v84, (__int64)&v105, &v98, &v97);
                  v22 = v97;
                  v57 = v89;
                }
                else
                {
                  sub_1F40D10((__int64)&v105, v19, v21, v103, v104);
                  v62 = v71;
                  LOBYTE(v62) = v105.m128i_i8[8];
                  v71 = v62;
                  v63 = sub_1D5E9F0(v19, v21, (unsigned int)v62, v106.m128i_i64[0]);
                  v57 = v89;
                  v22 = v63;
                }
              }
            }
          }
          v82 = a1[14];
          v75 = **v2;
        }
        v100 = v57;
        v90 = v57;
        LOBYTE(v99) = 0;
        if ( sub_1F58D20((__int64)&v99) )
        {
          v105.m128i_i8[0] = 0;
          LOBYTE(v101) = 0;
          v105.m128i_i64[1] = 0;
          v23 = sub_1F426C0(v82, v75, (unsigned int)v99, v90, (__int64)&v105, (unsigned int *)&v103, &v101);
          v18 = v90;
        }
        else
        {
          v91 = sub_1F58D40((__int64)&v99);
          v101 = v99;
          v73 = v99;
          v102 = v100;
          v74 = v100;
          if ( sub_1F58D20((__int64)&v101) )
          {
            v105.m128i_i8[0] = 0;
            v105.m128i_i64[1] = 0;
            LOBYTE(v98) = 0;
            sub_1F426C0(v82, v75, (unsigned int)v101, v102, (__int64)&v105, (unsigned int *)&v103, &v98);
            v59 = v98;
          }
          else
          {
            sub_1F40D10((__int64)&v105, v82, v75, v73, v74);
            v58 = v106.m128i_i64[0];
            LOBYTE(v103) = v105.m128i_i8[8];
            v104 = v106.m128i_i64[0];
            if ( v105.m128i_i8[8] )
            {
              v59 = *(_BYTE *)(v82 + v105.m128i_u8[8] + 1155);
            }
            else if ( sub_1F58D20((__int64)&v103) )
            {
              v105.m128i_i8[0] = 0;
              v105.m128i_i64[1] = 0;
              v97 = 0;
              sub_1F426C0(v82, v75, (unsigned int)v103, v58, (__int64)&v105, &v98, &v97);
              v59 = v97;
            }
            else
            {
              sub_1F40D10((__int64)&v105, v82, v75, v103, v104);
              v67 = v70;
              LOBYTE(v67) = v105.m128i_i8[8];
              v70 = v67;
              v59 = sub_1D5E9F0(v82, v75, (unsigned int)v67, v106.m128i_i64[0]);
            }
          }
          v60 = sub_1FD3510(v59);
          v23 = (v60 + v91 - 1) / v60;
        }
      }
      if ( v23 )
      {
        v24 = *((unsigned int *)v2 + 116);
        v25 = v22;
        v26 = 0;
        v27 = v18;
        do
        {
          v28 = *((_BYTE *)v2 + 8);
          v105.m128i_i8[8] = v25;
          v105.m128i_i64[0] = 0;
          v106.m128i_i8[0] = v17;
          v106.m128i_i64[1] = v27;
          v107.m128i_i8[0] = (v28 & 0x20) != 0;
          if ( (v28 & 1) != 0 )
            v105.m128i_i8[0] |= 2u;
          if ( (v28 & 2) != 0 )
            v105.m128i_i8[0] |= 1u;
          if ( (v28 & 8) != 0 )
            v105.m128i_i8[0] |= 4u;
          if ( (unsigned int)v24 >= *((_DWORD *)v2 + 117) )
          {
            v81 = v25;
            v86 = v23;
            sub_16CD150((__int64)(v2 + 57), v2 + 59, 0, 48, v25, v14);
            v24 = *((unsigned int *)v2 + 116);
            v25 = v81;
            v23 = v86;
          }
          ++v26;
          v29 = (__m128i *)&v2[57][6 * v24];
          *v29 = _mm_loadu_si128(&v105);
          v29[1] = _mm_loadu_si128(&v106);
          v29[2] = _mm_loadu_si128(&v107);
          v24 = (unsigned int)(*((_DWORD *)v2 + 116) + 1);
          *((_DWORD *)v2 + 116) = v24;
        }
        while ( v26 != v23 );
      }
      v93 += 16;
    }
    while ( v77 != v93 );
  }
  v30 = v2[6];
  v31 = v2[5];
  *((_DWORD *)v2 + 24) = 0;
  *((_DWORD *)v2 + 60) = 0;
  *((_DWORD *)v2 + 96) = 0;
  v79 = v30;
  if ( v31 != v30 )
  {
    v32 = v2;
    v33 = v31;
    v78 = (__int64)(v2 + 11);
    v76 = (__int64)(v2 + 29);
    v34 = v72;
    while ( 1 )
    {
      v44 = *((_BYTE *)v33 + 32);
      v45 = v33[3];
      v46 = v45;
      v47 = (v44 & 0x20) != 0;
      if ( (v44 & 0x20) != 0 )
        v46 = *(_QWORD *)(v45 + 24);
      v48 = a1[14];
      v49 = *(__int64 (**)())(*(_QWORD *)v48 + 1272LL);
      v50 = 0;
      if ( v49 != sub_1FD3430 )
      {
        v61 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, bool))v49)(
                v48,
                v46,
                *((unsigned int *)v32 + 4),
                ((_BYTE)v32[1] & 4) != 0);
        v45 = v33[3];
        v50 = v61;
        v44 = *((_BYTE *)v33 + 32);
        v47 = (v44 & 0x20) != 0;
      }
      v35 = ((unsigned __int64)((*((_BYTE *)v33 + 33) & 2) != 0) << 11)
          | (8LL * ((v44 & 8) != 0))
          | (4LL * ((v44 & 4) != 0))
          | (2LL * (v44 & 1))
          | ((v44 & 2) != 0)
          | ((unsigned __int64)(*((_BYTE *)v33 + 33) & 1) << 10)
          | v34 & 0xF8000010;
      if ( v47 )
        break;
      v35 &= ~0x10uLL;
      if ( (v44 & 0x40) != 0 )
        goto LABEL_40;
LABEL_27:
      v36 = v35 & 0xFDFFFFDF | ((32 * ((v44 & 0x10) != 0)) | (v50 << 25)) & 0x2000020 | v35 & 0xFFFFFFFF00000000LL;
      v37 = sub_15A9FE0(a1[12], v45);
      v40 = 0;
      if ( v37 )
      {
        _BitScanReverse(&v37, v37);
        v40 = (31 - (v37 ^ 0x1F) + 1) & 0x1F;
      }
      v41 = *((unsigned int *)v32 + 24);
      v42 = v36 & 0xFFFFFFFFFF07FFFFLL | ((unsigned __int64)(v40 & 0x1F) << 19);
      v34 = v42;
      if ( (unsigned int)v41 >= *((_DWORD *)v32 + 25) )
      {
        v95 = v42;
        sub_16CD150(v78, v32 + 13, 0, 8, v38, v39);
        v41 = *((unsigned int *)v32 + 24);
        v42 = v95;
      }
      v32[11][v41] = *v33;
      v43 = *((unsigned int *)v32 + 60);
      ++*((_DWORD *)v32 + 24);
      if ( (unsigned int)v43 >= *((_DWORD *)v32 + 61) )
      {
        v96 = v42;
        sub_16CD150(v76, v32 + 31, 0, 8, v38, v39);
        v43 = *((unsigned int *)v32 + 60);
        v42 = v96;
      }
      v33 += 5;
      v32[29][v43] = v42;
      ++*((_DWORD *)v32 + 60);
      if ( v79 == v33 )
      {
        v2 = v32;
        goto LABEL_62;
      }
    }
    LODWORD(v35) = v35 | 0x10;
    if ( (v44 & 0x40) != 0 )
LABEL_40:
      LOWORD(v35) = v35 | 0x110;
    v80 = v50;
    v85 = *(_QWORD *)(v45 + 24);
    v87 = a1[12];
    v94 = (unsigned int)sub_15A9FE0(v87, v85);
    v51 = sub_127FA20(v87, v85);
    v50 = v80;
    v52 = (v94 + ((unsigned __int64)(v51 + 7) >> 3) - 1) / v94;
    v53 = *((unsigned __int16 *)v33 + 17);
    if ( *((_WORD *)v33 + 17) )
    {
      v56 = ((v52 * v94) << 32) | (unsigned int)v35;
    }
    else
    {
      v88 = v52 * v94;
      v54 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(*(_QWORD *)a1[14] + 376LL))(a1[14], v85, a1[12]);
      v50 = v80;
      v53 = v54;
      v55 = 0;
      v56 = (v88 << 32) | (unsigned int)v35;
      if ( !v53 )
      {
LABEL_44:
        v45 = v33[3];
        v35 = ((unsigned __int64)(v55 & 0xF) << 15) | v56 & 0xFFFFFFFFFFF87FFFLL;
        v44 = *((_BYTE *)v33 + 32);
        goto LABEL_27;
      }
    }
    _BitScanReverse(&v53, v53);
    v55 = (31 - (v53 ^ 0x1F) + 1) & 0xF;
    goto LABEL_44;
  }
LABEL_62:
  v64 = *(__int64 (**)())(*a1 + 40LL);
  if ( v64 != sub_1FD34A0 && (v65 = ((__int64 (__fastcall *)(_QWORD *, __int64 **))v64)(a1, v2), (_BYTE)v65) )
  {
    sub_1E1B900((__int64)v2[9], (int *)v2[83], *((unsigned int *)v2 + 168), a1[15]);
    v68 = *((_DWORD *)v2 + 21);
    if ( v68 )
    {
      v69 = v2[8];
      if ( v69 )
        sub_1FD5CC0((__int64)a1, *v69 & 0xFFFFFFFFFFFFFFF8LL, *((_DWORD *)v2 + 20), v68);
    }
  }
  else
  {
LABEL_63:
    v65 = 0;
  }
  if ( (_BYTE *)v111[0] != v112 )
    _libc_free(v111[0]);
  if ( v108 != v110 )
    _libc_free((unsigned __int64)v108);
  return v65;
}
