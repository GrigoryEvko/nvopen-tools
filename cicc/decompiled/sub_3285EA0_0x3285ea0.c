// Function: sub_3285EA0
// Address: 0x3285ea0
//
__int64 __fastcall sub_3285EA0(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, int a6, __int128 a7)
{
  unsigned __int16 *v8; // rax
  __int64 v9; // r10
  __int64 v11; // rax
  int v15; // r15d
  __int64 v16; // rsi
  __m128i v17; // xmm0
  __int64 v18; // rsi
  __int64 v19; // rdi
  char v20; // al
  int v21; // r11d
  __int64 v22; // r10
  char v23; // r8
  __int64 v24; // rdi
  __int64 (__fastcall *v25)(__int64, __int64, __int64, int); // rax
  __int64 v26; // rax
  int v27; // edx
  bool v28; // zf
  int v29; // eax
  __int64 v30; // rdi
  char v31; // al
  __int64 v32; // rdi
  __int64 (__fastcall *v33)(__int64, __int64, __int64, int); // rax
  __int64 v34; // rax
  int v35; // edx
  char v36; // dl
  __m128i v37; // xmm3
  __m128i si128; // xmm2
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rdx
  char v46; // al
  int v47; // r9d
  __int64 v48; // rdi
  __int64 v49; // rax
  __m128i v50; // xmm5
  __m128i v51; // xmm4
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // rdi
  __m128i v57; // xmm7
  __int64 v58; // rax
  __int64 v59; // rdx
  char v60; // al
  int v61; // eax
  int v62; // edx
  int v63; // ecx
  __int64 v64; // r11
  __int64 v65; // r12
  __int64 v66; // rdx
  __int64 v67; // r13
  __int64 v68; // rdi
  __m128i v69; // xmm6
  __m128i v70; // xmm7
  __int128 v71; // rax
  int v72; // ecx
  char v73; // dl
  __int64 v74; // rsi
  __int64 v75; // r15
  __int32 v76; // eax
  __int128 v77; // rax
  int v78; // r10d
  __int64 v79; // rsi
  char v80; // al
  __int64 v81; // rsi
  char v82; // al
  __int64 v83; // r12
  __int64 v84; // rdx
  __int64 v85; // r13
  __int64 v86; // rdi
  __int128 v87; // [rsp-20h] [rbp-100h]
  __int128 v88; // [rsp-20h] [rbp-100h]
  __int128 v89; // [rsp-10h] [rbp-F0h]
  __int64 v90; // [rsp+8h] [rbp-D8h]
  int v91; // [rsp+10h] [rbp-D0h]
  int v92; // [rsp+10h] [rbp-D0h]
  __int64 v93; // [rsp+10h] [rbp-D0h]
  __int64 v94; // [rsp+10h] [rbp-D0h]
  int v95; // [rsp+18h] [rbp-C8h]
  __int64 v96; // [rsp+18h] [rbp-C8h]
  __int64 v97; // [rsp+18h] [rbp-C8h]
  __int64 v98; // [rsp+20h] [rbp-C0h]
  __int64 v99; // [rsp+28h] [rbp-B8h]
  __int128 v100; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v101; // [rsp+44h] [rbp-9Ch]
  __int64 v102; // [rsp+48h] [rbp-98h]
  __int128 v103; // [rsp+50h] [rbp-90h] BYREF
  __int64 v104; // [rsp+60h] [rbp-80h]
  __int64 v105; // [rsp+68h] [rbp-78h]
  __int128 v106; // [rsp+70h] [rbp-70h]
  __int64 v107; // [rsp+80h] [rbp-60h]
  __int64 v108; // [rsp+88h] [rbp-58h]
  __m128i v109; // [rsp+90h] [rbp-50h] BYREF
  __m128i v110; // [rsp+A0h] [rbp-40h]

  v107 = a3;
  LODWORD(v105) = a6;
  *(_QWORD *)&v106 = a7;
  v8 = (unsigned __int16 *)(*(_QWORD *)(a4 + 48) + 16LL * (unsigned int)a5);
  v9 = *v8;
  v108 = *((_QWORD *)v8 + 1);
  if ( a2 != *(_DWORD *)(a4 + 24) )
    return 0;
  v11 = *(_QWORD *)(a4 + 40);
  v102 = v9;
  v15 = a5;
  v16 = *(_QWORD *)v11;
  LODWORD(v104) = DWORD2(a7);
  v17 = _mm_loadu_si128((const __m128i *)v11);
  v98 = v16;
  LODWORD(v16) = *(_DWORD *)(v11 + 8);
  v103 = (__int128)_mm_loadu_si128((const __m128i *)(v11 + 40));
  v101 = v16;
  v18 = *(_QWORD *)(v11 + 40);
  LODWORD(v11) = *(_DWORD *)(v11 + 48);
  v19 = *a1;
  v100 = (__int128)v17;
  v99 = v18;
  v95 = v11;
  v20 = sub_33E2390(v19, v103, *((_QWORD *)&v103 + 1), 1);
  v21 = v104;
  v22 = v102;
  v23 = v20;
  if ( !v20 )
    goto LABEL_4;
  v28 = *(_DWORD *)(a4 + 24) == 56;
  LODWORD(v104) = 0;
  if ( v28 )
  {
    v29 = v105 & 1;
    if ( (*(_BYTE *)(a4 + 28) & 1) == 0 )
      v29 = v104;
    LODWORD(v104) = v29;
  }
  v90 = v102;
  v30 = *a1;
  v91 = v21;
  LOBYTE(v102) = v23;
  v31 = sub_33E2390(v30, a7, *((_QWORD *)&a7 + 1), 1);
  v21 = v91;
  v22 = v90;
  if ( !v31 )
  {
    v32 = a1[1];
    v33 = *(__int64 (__fastcall **)(__int64, __int64, __int64, int))(*(_QWORD *)v32 + 1864LL);
    if ( v33 != sub_302F6C0 )
    {
      v81 = *a1;
      LODWORD(v102) = v91;
      v82 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))v33)(
              v32,
              v81,
              a4,
              a5,
              a7,
              *((_QWORD *)&a7 + 1));
      v22 = v90;
      v21 = v91;
      v73 = v82;
      goto LABEL_66;
    }
    v34 = *(_QWORD *)(a4 + 56);
    v35 = 1;
    if ( !v34 )
      goto LABEL_4;
    while ( 1 )
    {
      if ( *(_DWORD *)(v34 + 8) == v15 )
      {
        if ( !v35 )
          goto LABEL_4;
        v34 = *(_QWORD *)(v34 + 32);
        if ( !v34 )
        {
          v73 = v102;
LABEL_66:
          if ( v73 )
          {
            v74 = *(_QWORD *)(a4 + 80);
            v75 = *a1;
            v109.m128i_i64[0] = v74;
            if ( v74 )
            {
              *(_QWORD *)&v106 = v22;
              sub_B96E90((__int64)&v109, v74, 1);
              v22 = v106;
            }
            v76 = *(_DWORD *)(a4 + 72);
            *(_QWORD *)&v106 = v22;
            v109.m128i_i32[2] = v76;
            *(_QWORD *)&v77 = sub_3405C90(v75, a2, (unsigned int)&v109, v22, v108, v104, v100, a7);
            v78 = v106;
            if ( v109.m128i_i64[0] )
            {
              v105 = v106;
              v106 = v77;
              sub_B91220((__int64)&v109, v109.m128i_i64[0]);
              v78 = v105;
              v77 = v106;
            }
            return sub_3405C90(*a1, a2, v107, v78, v108, v104, v77, v103);
          }
LABEL_4:
          LODWORD(v104) = a2 - 186;
          if ( a2 - 186 <= 1 )
          {
            if ( v101 == v21 && v98 == (_QWORD)v106 || v21 == v95 && v99 == (_QWORD)v106 )
              return a4;
          }
          else if ( a2 == 188 )
          {
            if ( v101 == v21 && v98 == (_QWORD)v106 )
              return v103;
            if ( v99 == (_QWORD)v106 && v21 == v95 )
              return v100;
          }
          v24 = a1[1];
          v25 = *(__int64 (__fastcall **)(__int64, __int64, __int64, int))(*(_QWORD *)v24 + 1864LL);
          if ( v25 == sub_302F6C0 )
          {
            v26 = *(_QWORD *)(a4 + 56);
            v27 = 1;
            if ( !v26 )
              return 0;
            do
            {
              if ( *(_DWORD *)(v26 + 8) == v15 )
              {
                if ( !v27 )
                  return 0;
                v26 = *(_QWORD *)(v26 + 32);
                if ( !v26 )
                  goto LABEL_31;
                if ( *(_DWORD *)(v26 + 8) == v15 )
                  return 0;
                v27 = 0;
              }
              v26 = *(_QWORD *)(v26 + 32);
            }
            while ( v26 );
            v36 = v27 ^ 1;
          }
          else
          {
            v94 = v22;
            v79 = *a1;
            LODWORD(v102) = v21;
            v80 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))v25)(
                    v24,
                    v79,
                    a4,
                    a5,
                    a7,
                    *((_QWORD *)&a7 + 1));
            v22 = v94;
            v21 = v102;
            v36 = v80;
          }
          if ( v36 )
          {
LABEL_31:
            if ( v99 != (_QWORD)v106 || v21 != v95 )
            {
              v37 = _mm_loadu_si128((const __m128i *)&a7);
              si128 = _mm_load_si128((const __m128i *)&v100);
              v92 = v21;
              v96 = v22;
              v102 = *a1;
              v109 = si128;
              v110 = v37;
              v39 = sub_33ED250(v102, (unsigned int)v22, v108, v106);
              v41 = sub_33D01C0(v102, a2, v39, v40, &v109, 2);
              v22 = v96;
              v21 = v92;
              v102 = v41;
              if ( v41 )
              {
                v43 = *a1;
                v109.m128i_i64[0] = v41;
                v110 = _mm_load_si128((const __m128i *)&v103);
                v109.m128i_i32[2] = 0;
                v44 = sub_33ED250(v43, (unsigned int)v96, v108, v42);
                v46 = sub_33CEDC0(v43, a2, v44, v45, &v109, 2);
                v22 = v96;
                v21 = v92;
                if ( !v46 )
                {
                  v48 = *a1;
                  v49 = v102;
                  v89 = v103;
                  return sub_3406EB0(v48, a2, v107, v22, v108, v47, (unsigned __int64)v49, v89);
                }
              }
            }
            if ( v98 != (_QWORD)v106 || v101 != v21 )
            {
              v50 = _mm_loadu_si128((const __m128i *)&a7);
              v51 = _mm_load_si128((const __m128i *)&v103);
              v97 = v22;
              v102 = *a1;
              v109 = v51;
              v110 = v50;
              v52 = sub_33ED250(v102, (unsigned int)v22, v108, v106);
              v54 = sub_33D01C0(v102, a2, v52, v53, &v109, 2);
              v22 = v97;
              if ( v54 )
              {
                v56 = *a1;
                v109.m128i_i64[0] = v54;
                v57 = _mm_load_si128((const __m128i *)&v100);
                v93 = v54;
                v102 = v56;
                v110 = v57;
                v109.m128i_i32[2] = 0;
                v58 = sub_33ED250(v56, (unsigned int)v97, v108, v55);
                v60 = sub_33CEDC0(v102, a2, v58, v59, &v109, 2);
                v22 = v97;
                if ( !v60 )
                {
                  v48 = *a1;
                  v49 = v93;
                  v89 = v100;
                  return sub_3406EB0(v48, a2, v107, v22, v108, v47, (unsigned __int64)v49, v89);
                }
              }
            }
            if ( (unsigned int)v104 > 1
              || *(_DWORD *)(v106 + 24) != 208
              || *(_DWORD *)(v98 + 24) != 208
              || *(_DWORD *)(v99 + 24) != 208 )
            {
              return 0;
            }
            v61 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v106 + 40) + 80LL) + 96LL);
            v62 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v98 + 40) + 80LL) + 96LL);
            v63 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v99 + 40) + 80LL) + 96LL);
            if ( v62 == v61 )
            {
              if ( v63 != v61 )
              {
                v64 = *a1;
                *(_QWORD *)&v106 = v22;
                v104 = v64;
                sub_3285E70((__int64)&v109, a4);
                *(_QWORD *)&v100 = v98;
                *((_QWORD *)&v100 + 1) = v101 | *((_QWORD *)&v100 + 1) & 0xFFFFFFFF00000000LL;
                v65 = sub_3405C90(
                        v104,
                        a2,
                        (unsigned int)&v109,
                        v106,
                        v108,
                        v105,
                        __PAIR128__(*((unsigned __int64 *)&v100 + 1), v98),
                        a7);
                v67 = v66;
                sub_9C6650(&v109);
                *((_QWORD *)&v87 + 1) = v67;
                *(_QWORD *)&v87 = v65;
                return sub_3405C90(*a1, a2, v107, v106, v108, v105, v87, v103);
              }
              goto LABEL_78;
            }
            if ( v63 == v61 )
            {
LABEL_78:
              *(_QWORD *)&v106 = v22;
              if ( v62 != v61 )
              {
                v104 = *a1;
                sub_3285E70((__int64)&v109, a4);
                v83 = sub_3405C90(v104, a2, (unsigned int)&v109, v106, v108, v105, v103, a7);
                v85 = v84;
                sub_9C6650(&v109);
                v86 = *a1;
                *(_QWORD *)&v100 = v98;
                *((_QWORD *)&v88 + 1) = v85;
                *(_QWORD *)&v88 = v83;
                return sub_3405C90(
                         v86,
                         a2,
                         v107,
                         v106,
                         v108,
                         v105,
                         v88,
                         __PAIR128__(v101 | *((_QWORD *)&v100 + 1) & 0xFFFFFFFF00000000LL, v98));
              }
            }
          }
          return 0;
        }
        if ( *(_DWORD *)(v34 + 8) == v15 )
          goto LABEL_4;
        v35 = 0;
      }
      v34 = *(_QWORD *)(v34 + 32);
      if ( !v34 )
      {
        v73 = v35 ^ 1;
        goto LABEL_66;
      }
    }
  }
  v68 = *a1;
  v69 = _mm_load_si128((const __m128i *)&v103);
  v70 = _mm_loadu_si128((const __m128i *)&a7);
  *(_QWORD *)&v106 = v90;
  v109 = v69;
  v110 = v70;
  *(_QWORD *)&v71 = sub_3402EA0(v68, a2, v107, v90, v108, 0, (__int64)&v109, 2);
  if ( !(_QWORD)v71 )
    return 0;
  if ( (v105 & 8) != 0 )
  {
    v72 = v104 | 8;
    if ( (*(_BYTE *)(a4 + 28) & 8) == 0 )
      v72 = v104;
    LODWORD(v104) = v72;
  }
  return sub_3405C90(*a1, a2, v107, v106, v108, v104, v100, v71);
}
