// Function: sub_2040C70
// Address: 0x2040c70
//
__int64 __fastcall sub_2040C70(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  unsigned int v5; // ebx
  unsigned int *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned __int8 v10; // r14
  __int64 v11; // r13
  unsigned __int8 *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  char v16; // al
  __int64 v17; // rsi
  __int64 v18; // r9
  unsigned int v19; // r13d
  unsigned int v20; // ecx
  unsigned int v21; // r15d
  unsigned int v22; // eax
  int v23; // r8d
  __int64 v24; // r9
  _QWORD *v25; // rax
  const void **v26; // rdx
  _QWORD *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r13
  char v30; // al
  __int64 v31; // rsi
  __int128 v32; // rax
  __int64 **v33; // r12
  int v34; // edx
  __int64 *v35; // r12
  __int64 v36; // rax
  unsigned int v37; // edx
  _QWORD *v38; // rbx
  int v39; // edx
  int v40; // r12d
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 *v43; // rax
  unsigned __int64 v44; // rdi
  __int64 v45; // r12
  unsigned int v47; // edx
  __int64 v48; // rcx
  int v49; // r8d
  int v50; // r9d
  _QWORD *v51; // r13
  int v52; // edx
  int v53; // r14d
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rcx
  unsigned int i; // edx
  __int64 v59; // rax
  _QWORD *v60; // rax
  int v61; // r8d
  int v62; // r9d
  int v63; // edx
  __int64 v64; // rsi
  __int64 v65; // rax
  _QWORD *v66; // rdi
  __int64 v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rcx
  int v70; // esi
  const void *v71; // r14
  __int64 v72; // r15
  __int64 v73; // rbx
  __int64 v74; // r12
  __int64 v75; // rdx
  __int64 v76; // r13
  __int64 v77; // rax
  __int64 v78; // rdx
  _QWORD *v79; // rax
  __int128 v80; // [rsp-10h] [rbp-250h]
  __int128 v81; // [rsp-10h] [rbp-250h]
  unsigned int v82; // [rsp+14h] [rbp-22Ch]
  __int64 v84; // [rsp+20h] [rbp-220h]
  __int64 v86; // [rsp+30h] [rbp-210h]
  unsigned int v87; // [rsp+38h] [rbp-208h]
  const void **v88; // [rsp+40h] [rbp-200h]
  __m128i v89; // [rsp+50h] [rbp-1F0h]
  int v90; // [rsp+60h] [rbp-1E0h]
  unsigned int v91; // [rsp+64h] [rbp-1DCh]
  __int64 v92; // [rsp+68h] [rbp-1D8h]
  __int64 (__fastcall *v93)(__int64, __int64); // [rsp+70h] [rbp-1D0h]
  char v94; // [rsp+7Bh] [rbp-1C5h]
  unsigned int v95; // [rsp+7Ch] [rbp-1C4h]
  __int64 *v96; // [rsp+90h] [rbp-1B0h]
  __int64 v97; // [rsp+C0h] [rbp-180h] BYREF
  __int64 v98; // [rsp+C8h] [rbp-178h]
  __int64 v99; // [rsp+D0h] [rbp-170h] BYREF
  const void **v100; // [rsp+D8h] [rbp-168h]
  __int64 v101; // [rsp+E0h] [rbp-160h] BYREF
  int v102; // [rsp+E8h] [rbp-158h]
  __m128i v103; // [rsp+F0h] [rbp-150h] BYREF
  _QWORD *v104; // [rsp+100h] [rbp-140h] BYREF
  __int64 v105; // [rsp+108h] [rbp-138h]
  _QWORD s[38]; // [rsp+110h] [rbp-130h] BYREF

  v7 = *(unsigned int **)(a2 + 32);
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * v7[2];
  v10 = *(_BYTE *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v12 = *(unsigned __int8 **)(a2 + 40);
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL);
  LOBYTE(v97) = v10;
  v14 = *v12;
  v15 = *((_QWORD *)v12 + 1);
  v98 = v11;
  sub_1F40D10((__int64)&v104, v8, v13, v14, v15);
  v16 = v105;
  v17 = *(_QWORD *)(a2 + 72);
  LOBYTE(v99) = v105;
  v100 = (const void **)s[0];
  v101 = v17;
  if ( v17 )
  {
    sub_1623A60((__int64)&v101, v17, 2);
    v16 = v99;
  }
  v102 = *(_DWORD *)(a2 + 64);
  if ( v16 )
  {
    v82 = word_4305480[(unsigned __int8)(v16 - 14)];
    if ( v10 )
    {
LABEL_5:
      v90 = word_4305480[(unsigned __int8)(v10 - 14)];
      goto LABEL_6;
    }
  }
  else
  {
    v82 = sub_1F58D30((__int64)&v99);
    if ( v10 )
      goto LABEL_5;
  }
  v90 = sub_1F58D30((__int64)&v97);
LABEL_6:
  v91 = *(_DWORD *)(a2 + 56);
  sub_1F40D10((__int64)&v104, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v10, v11);
  if ( (_BYTE)v104 != 7 )
  {
    if ( (_BYTE)v99 )
    {
      v19 = word_4305480[(unsigned __int8)(v99 - 14)];
      if ( v10 )
        goto LABEL_9;
    }
    else
    {
      v19 = sub_1F58D30((__int64)&v99);
      if ( v10 )
      {
LABEL_9:
        v20 = word_4305480[(unsigned __int8)(v10 - 14)];
        goto LABEL_10;
      }
    }
    v20 = sub_1F58D30((__int64)&v97);
LABEL_10:
    v94 = 0;
    v21 = v19 / v20;
    if ( !(v19 % v20) )
    {
      v104 = 0;
      LODWORD(v105) = 0;
      v51 = sub_1D2B300(*(_QWORD **)(a1 + 8), 0x30u, (__int64)&v104, v97, v98, v18);
      v53 = v52;
      if ( v104 )
        sub_161E7C0((__int64)&v104, (__int64)v104);
      v104 = s;
      v103.m128i_i64[0] = 0;
      v103.m128i_i32[2] = 0;
      v105 = 0x1000000000LL;
      sub_202F910((__int64)&v104, v21, &v103, v48, v49, v50);
      if ( v91 )
      {
        v54 = 0;
        v55 = 0;
        do
        {
          v56 = *(_QWORD *)(a2 + 32);
          v57 = (__int64)v104;
          v104[v54] = *(_QWORD *)(v56 + v55);
          LODWORD(v56) = *(_DWORD *)(v56 + v55 + 8);
          v55 += 40;
          *(_DWORD *)(v57 + v54 * 8 + 8) = v56;
          v54 += 2;
        }
        while ( v55 != 40LL * v91 );
      }
      for ( i = v91; v21 != i; *((_DWORD *)v60 + 2) = v53 )
      {
        v59 = i++;
        v60 = &v104[2 * v59];
        *v60 = v51;
      }
      *((_QWORD *)&v81 + 1) = (unsigned int)v105;
      *(_QWORD *)&v81 = v104;
      v43 = sub_1D359D0(
              *(__int64 **)(a1 + 8),
              107,
              (__int64)&v101,
              (unsigned int)v99,
              v100,
              0,
              *(double *)a3.m128i_i64,
              a4,
              a5,
              v81);
LABEL_46:
      v44 = (unsigned __int64)v104;
      v45 = (__int64)v43;
      if ( v104 == s )
        goto LABEL_48;
      goto LABEL_47;
    }
LABEL_11:
    LOBYTE(v22) = sub_1F7E0F0((__int64)&v99);
    v87 = v22;
    v25 = s;
    v88 = v26;
    v104 = s;
    v105 = 0x1000000000LL;
    if ( v82 > 0x10 )
    {
      sub_16CD150((__int64)&v104, s, v82, 16, v23, v24);
      v25 = v104;
    }
    v27 = &v25[2 * v82];
    for ( LODWORD(v105) = v82; v27 != v25; v25 += 2 )
    {
      if ( v25 )
      {
        *v25 = 0;
        *((_DWORD *)v25 + 2) = 0;
      }
    }
    if ( v91 )
    {
      v28 = v91;
      v86 = 0;
      v91 = 0;
      v84 = 40 * v28;
      do
      {
        a3 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 32) + v86));
        v89 = a3;
        if ( v94 )
        {
          v89.m128i_i64[0] = sub_20363F0(a1, a3.m128i_u64[0], a3.m128i_i64[1]);
          v89.m128i_i64[1] = v47 | a3.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        }
        if ( v90 )
        {
          v29 = 0;
          do
          {
            v35 = *(__int64 **)(a1 + 8);
            v92 = *(_QWORD *)a1;
            v95 = v29 + v91;
            v93 = *(__int64 (__fastcall **)(__int64, __int64))(**(_QWORD **)a1 + 48LL);
            v36 = sub_1E0A0C0(v35[4]);
            if ( v93 == sub_1D13A20 )
            {
              v37 = 8 * sub_15A9520(v36, 0);
              if ( v37 == 32 )
              {
                v30 = 5;
              }
              else if ( v37 <= 0x20 )
              {
                v30 = 3;
                if ( v37 != 8 )
                  v30 = 4 * (v37 == 16);
              }
              else
              {
                v30 = 6;
                if ( v37 != 64 )
                {
                  v30 = 0;
                  if ( v37 == 128 )
                    v30 = 7;
                }
              }
            }
            else
            {
              v30 = v93(v92, v36);
            }
            LOBYTE(v5) = v30;
            v31 = v29++;
            *(_QWORD *)&v32 = sub_1D38BB0((__int64)v35, v31, (__int64)&v101, v5, 0, 0, a3, a4, a5, 0);
            v96 = sub_1D332F0(
                    v35,
                    106,
                    (__int64)&v101,
                    v87,
                    v88,
                    0,
                    *(double *)a3.m128i_i64,
                    a4,
                    a5,
                    v89.m128i_i64[0],
                    v89.m128i_u64[1],
                    v32);
            v33 = (__int64 **)&v104[2 * v95];
            *v33 = v96;
            *((_DWORD *)v33 + 2) = v34;
          }
          while ( v90 != v29 );
          v91 += v90;
        }
        v86 += 40;
      }
      while ( v84 != v86 );
    }
    v103.m128i_i64[0] = 0;
    v103.m128i_i32[2] = 0;
    v38 = sub_1D2B300(*(_QWORD **)(a1 + 8), 0x30u, (__int64)&v103, v87, (__int64)v88, v24);
    v40 = v39;
    if ( v103.m128i_i64[0] )
      sub_161E7C0((__int64)&v103, v103.m128i_i64[0]);
    if ( v91 < v82 )
    {
      v41 = 2LL * v91;
      do
      {
        v42 = (__int64)v104;
        v104[v41] = v38;
        *(_DWORD *)(v42 + v41 * 8 + 8) = v40;
        v41 += 2;
      }
      while ( 2 * (v91 + (unsigned __int64)(v82 - 1 - v91) + 1) != v41 );
    }
    *((_QWORD *)&v80 + 1) = (unsigned int)v105;
    *(_QWORD *)&v80 = v104;
    v43 = sub_1D359D0(*(__int64 **)(a1 + 8), 104, (__int64)&v101, v99, v100, 0, *(double *)a3.m128i_i64, a4, a5, v80);
    goto LABEL_46;
  }
  sub_1F40D10((__int64)&v104, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v97, v98);
  if ( (_BYTE)v99 != (_BYTE)v105 || !(_BYTE)v99 && (const void **)s[0] != v100 )
  {
LABEL_77:
    v94 = 1;
    goto LABEL_11;
  }
  if ( v91 <= 1 )
  {
    if ( v91 != 1 )
      goto LABEL_77;
    v64 = *(_QWORD *)(a2 + 32);
LABEL_80:
    v45 = sub_20363F0(a1, *(_QWORD *)v64, *(_QWORD *)(v64 + 8));
  }
  else
  {
    v63 = 1;
    v64 = *(_QWORD *)(a2 + 32);
    v65 = v64 + 40;
    while ( *(_WORD *)(*(_QWORD *)v65 + 24LL) == 48 )
    {
      ++v63;
      v65 += 40;
      if ( v63 == v91 )
        goto LABEL_80;
    }
    v94 = 1;
    if ( v91 != 2 )
      goto LABEL_11;
    v66 = s;
    v104 = s;
    v105 = 0x1000000000LL;
    if ( v82 > 0x10 )
    {
      sub_16CD150((__int64)&v104, s, v82, 4, v61, v62);
      v66 = v104;
    }
    v67 = (__int64)v66 + 4 * v82;
    LODWORD(v105) = v82;
    if ( v66 != (_QWORD *)v67 )
    {
      memset(v66, 255, 4LL * v82);
      v67 = (__int64)v104;
    }
    if ( v90 )
    {
      v68 = 0;
      do
      {
        *(_DWORD *)(v67 + 4 * v68) = v68;
        v69 = (unsigned int)(v90 + v68);
        v70 = v82 + v68++;
        *((_DWORD *)v104 + v69) = v70;
        v67 = (__int64)v104;
      }
      while ( v90 != v68 );
    }
    v71 = (const void *)v67;
    v72 = (unsigned int)v105;
    v73 = *(_QWORD *)(a1 + 8);
    v74 = sub_20363F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
    v76 = v75;
    v77 = sub_20363F0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
    v79 = sub_1D41320(
            v73,
            (unsigned int)v99,
            v100,
            (__int64)&v101,
            v77,
            v78,
            *(double *)a3.m128i_i64,
            a4,
            a5,
            v74,
            v76,
            v71,
            v72);
    v44 = (unsigned __int64)v104;
    v45 = (__int64)v79;
    if ( v104 != s )
LABEL_47:
      _libc_free(v44);
  }
LABEL_48:
  if ( v101 )
    sub_161E7C0((__int64)&v101, v101);
  return v45;
}
