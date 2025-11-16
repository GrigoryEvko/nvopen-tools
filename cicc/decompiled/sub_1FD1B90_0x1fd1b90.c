// Function: sub_1FD1B90
// Address: 0x1fd1b90
//
void __fastcall sub_1FD1B90(const __m128i *a1, int a2, __int64 a3, unsigned int a4, double a5, __m128 a6, __m128i a7)
{
  __int64 v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // r12
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // r9
  __int64 v14; // r8
  __int64 v15; // rcx
  char v16; // si
  unsigned int v17; // eax
  __int64 v18; // rcx
  const __m128i *v19; // rax
  const __m128i *v20; // r12
  const __m128i *v21; // r13
  __int64 v22; // rsi
  __int128 v23; // rdi
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 *v26; // rcx
  __int64 v27; // r8
  _QWORD *v28; // r9
  __m128i v29; // xmm0
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // r12
  __int64 *v36; // rax
  __int64 v37; // rdi
  __int64 *v38; // rdi
  bool v39; // zf
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // rax
  __int64 *v43; // r8
  __int64 v44; // r9
  __int64 v45; // rcx
  __int64 *v46; // r13
  __int64 *v47; // r14
  __int64 v48; // r15
  __int64 v49; // rsi
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int64 v52; // rcx
  _QWORD *v53; // r8
  _QWORD *v54; // r9
  __int64 i; // r13
  __int64 v56; // rbx
  const __m128i *v57; // r12
  __int64 v58; // r13
  __int64 **v59; // rax
  __int8 v60; // al
  __int64 *v61; // r13
  __int64 *v62; // r15
  __int64 v63; // r14
  __int64 j; // r14
  __int64 *v65; // rcx
  __int64 *v66; // rdi
  int v67; // eax
  __m128i v68; // [rsp+20h] [rbp-5C0h] BYREF
  __int64 v69; // [rsp+30h] [rbp-5B0h]
  __int64 v70; // [rsp+38h] [rbp-5A8h]
  __int64 v71; // [rsp+40h] [rbp-5A0h]
  __int64 v72; // [rsp+48h] [rbp-598h]
  __m128i v73; // [rsp+50h] [rbp-590h]
  __int64 (__fastcall **v74)(); // [rsp+60h] [rbp-580h] BYREF
  __int64 v75; // [rsp+68h] [rbp-578h]
  __int64 *v76; // [rsp+70h] [rbp-570h]
  const __m128i **v77; // [rsp+78h] [rbp-568h]
  _QWORD v78[7]; // [rsp+80h] [rbp-560h] BYREF
  __int64 v79; // [rsp+B8h] [rbp-528h]
  int v80; // [rsp+C0h] [rbp-520h]
  __int64 v81; // [rsp+C8h] [rbp-518h]
  int v82; // [rsp+D0h] [rbp-510h]
  __int64 v83; // [rsp+D8h] [rbp-508h] BYREF
  __int64 v84; // [rsp+E0h] [rbp-500h]
  _QWORD *v85; // [rsp+E8h] [rbp-4F8h]
  __int64 v86; // [rsp+F0h] [rbp-4F0h]
  __int64 v87; // [rsp+F8h] [rbp-4E8h] BYREF
  __int64 *v88; // [rsp+100h] [rbp-4E0h] BYREF
  __int64 v89; // [rsp+108h] [rbp-4D8h]
  __int64 v90; // [rsp+110h] [rbp-4D0h] BYREF
  __int64 *v91; // [rsp+190h] [rbp-450h] BYREF
  __int64 v92; // [rsp+198h] [rbp-448h]
  _BYTE v93[128]; // [rsp+1A0h] [rbp-440h] BYREF
  const __m128i *v94; // [rsp+220h] [rbp-3C0h] BYREF
  __int64 v95; // [rsp+228h] [rbp-3B8h]
  int v96; // [rsp+230h] [rbp-3B0h]
  unsigned int v97; // [rsp+234h] [rbp-3ACh]
  __int16 v98; // [rsp+238h] [rbp-3A8h]
  char v99; // [rsp+23Ah] [rbp-3A6h]
  _BYTE *v100; // [rsp+240h] [rbp-3A0h]
  __int64 v101; // [rsp+248h] [rbp-398h]
  _BYTE v102[512]; // [rsp+250h] [rbp-390h] BYREF
  __int64 v103; // [rsp+450h] [rbp-190h]
  __int64 v104; // [rsp+458h] [rbp-188h]
  __int64 v105; // [rsp+460h] [rbp-180h]
  int v106; // [rsp+468h] [rbp-178h]
  __int64 v107; // [rsp+470h] [rbp-170h] BYREF
  __int64 *v108; // [rsp+478h] [rbp-168h]
  __int64 *v109; // [rsp+480h] [rbp-160h]
  __int64 v110; // [rsp+488h] [rbp-158h]
  int v111; // [rsp+490h] [rbp-150h]
  _BYTE v112[256]; // [rsp+498h] [rbp-148h] BYREF
  __int64 v113; // [rsp+598h] [rbp-48h]
  unsigned int v114; // [rsp+5A0h] [rbp-40h]

  v8 = a1[1].m128i_i64[0];
  v94 = a1;
  v98 = 0;
  v95 = v8;
  v100 = v102;
  v101 = 0x4000000000LL;
  v108 = (__int64 *)v112;
  v109 = (__int64 *)v112;
  v9 = (_QWORD *)a1[2].m128i_i64[0];
  v96 = 0;
  v97 = a4;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v110 = 32;
  v111 = 0;
  v113 = a3;
  v10 = *v9 + 112LL;
  v11 = sub_1560180(v10, 34);
  if ( !v11 )
    v11 = sub_1560180(v10, 17);
  v99 = v11;
  v14 = v95;
  v15 = 1;
  v114 = 0;
  while ( 1 )
  {
    v16 = v15;
    if ( v15 != 1 )
    {
      if ( *(_QWORD *)(v14 + 8 * v15 + 120) )
      {
        v17 = sub_1F6C8D0(v15);
        if ( v17 >= v114 )
          v114 = v17;
      }
      if ( v16 == 114 )
        break;
    }
    ++v15;
  }
  v18 = v97;
  if ( v97 )
  {
    v19 = v94;
    v96 = a2;
    LOBYTE(v98) = a2 > 1;
    v20 = (const __m128i *)v94[12].m128i_i64[1];
    v21 = v94 + 12;
    HIBYTE(v98) = a2 > 0;
    if ( &v94[12] != v20 )
    {
      do
      {
        v22 = (__int64)&v20[-1].m128i_i64[1];
        if ( !v20 )
          v22 = 0;
        sub_1F81BC0((__int64)&v94, v22);
        v20 = (const __m128i *)v20->m128i_i64[1];
      }
      while ( v21 != v20 );
      v19 = v94;
    }
    *((_QWORD *)&v23 + 1) = 0;
    v24 = v19[11].m128i_i64[0];
    v68 = _mm_loadu_si128(v19 + 11);
    v25 = sub_1D274F0(1u, v12, v18, v14, v13);
    v29 = _mm_load_si128(&v68);
    v87 = 0;
    v78[5] = v25;
    v79 = 0x100000000LL;
    v85 = v78;
    v73 = v29;
    v78[6] = 0;
    v80 = 0;
    v86 = 0;
    v81 = 0;
    v82 = -65536;
    LODWORD(v84) = v29.m128i_i32[2];
    v83 = v29.m128i_i64[0];
    v30 = *(_QWORD *)(v24 + 48);
    memset(v78, 0, 24);
    v78[3] = -4294967084LL;
    v87 = v30;
    if ( v30 )
    {
      v26 = &v87;
      *(_QWORD *)(v30 + 24) = &v87;
    }
    v86 = v24 + 48;
    *(_QWORD *)(v24 + 48) = &v83;
    v31 = (unsigned int)v105;
    v78[4] = &v83;
    LODWORD(v79) = 1;
    while ( (_DWORD)v105 )
    {
      while ( 1 )
      {
        v32 = (__int64)v100;
        v33 = v101;
        do
        {
          v34 = v33--;
          v35 = *(_QWORD *)&v100[8 * v34 - 8];
          LODWORD(v101) = v33;
        }
        while ( !v35 );
        if ( v106 )
        {
          v32 = v104;
          v34 = (v106 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
          v36 = (__int64 *)(v104 + 16 * v34);
          v37 = *v36;
          if ( v35 == *v36 )
          {
LABEL_23:
            *v36 = -16;
            LODWORD(v105) = v105 - 1;
            ++HIDWORD(v105);
          }
          else
          {
            v67 = 1;
            while ( v37 != -8 )
            {
              v27 = (unsigned int)(v67 + 1);
              v34 = (v106 - 1) & (unsigned int)(v67 + v34);
              v36 = (__int64 *)(v104 + 16LL * (unsigned int)v34);
              v37 = *v36;
              if ( v35 == *v36 )
                goto LABEL_23;
              v67 = v27;
            }
          }
        }
        *((_QWORD *)&v23 + 1) = v35;
        if ( (unsigned __int8)sub_1FC6430((__int64 *)&v94, v35, v34, v32, (_QWORD *)v27, v28) )
          goto LABEL_54;
        v38 = (__int64 *)v94;
        v39 = v96 == 3;
        v77 = &v94;
        v40 = v94[41].m128i_i64[1];
        v76 = (__int64 *)v94;
        v75 = v40;
        v94[41].m128i_i64[1] = (__int64)&v74;
        v74 = off_49FFF30;
        if ( v39 )
          break;
LABEL_26:
        v41 = (__int64)v109;
        v42 = v108;
        if ( v109 != v108 )
          goto LABEL_27;
        v43 = &v109[HIDWORD(v110)];
        v44 = HIDWORD(v110);
        if ( v109 == v43 )
        {
LABEL_94:
          if ( HIDWORD(v110) >= (unsigned int)v110 )
          {
LABEL_27:
            sub_16CCBA0((__int64)&v107, v35);
            v41 = (__int64)v109;
            v42 = v108;
            goto LABEL_28;
          }
          v44 = (unsigned int)++HIDWORD(v110);
          *v43 = v35;
          v42 = v108;
          ++v107;
          v41 = (__int64)v109;
        }
        else
        {
          v65 = v109;
          v66 = 0;
          while ( v35 != *v65 )
          {
            if ( *v65 == -2 )
              v66 = v65;
            if ( v43 == ++v65 )
            {
              if ( !v66 )
                goto LABEL_94;
              *v66 = v35;
              v41 = (__int64)v109;
              --v111;
              v42 = v108;
              ++v107;
              break;
            }
          }
        }
LABEL_28:
        v45 = *(_QWORD *)(v35 + 32);
        v68.m128i_i64[0] = v45 + 40LL * *(unsigned int *)(v35 + 56);
        if ( v45 != v68.m128i_i64[0] )
        {
          v46 = (__int64 *)v45;
          while ( 1 )
          {
            v48 = *v46;
            if ( (__int64 *)v41 == v42 )
            {
              v47 = &v42[HIDWORD(v110)];
              if ( v42 == v47 )
              {
                v41 = (__int64)v42;
              }
              else
              {
                do
                {
                  if ( v48 == *v42 )
                    break;
                  ++v42;
                }
                while ( v47 != v42 );
                v41 = (__int64)v47;
              }
LABEL_43:
              while ( (__int64 *)v41 != v42 )
              {
                if ( (unsigned __int64)*v42 < 0xFFFFFFFFFFFFFFFELL )
                {
                  if ( v47 != v42 )
                    goto LABEL_34;
                  goto LABEL_46;
                }
                ++v42;
              }
            }
            else
            {
              v47 = (__int64 *)(v41 + 8LL * (unsigned int)v110);
              v42 = sub_16CC9F0((__int64)&v107, *v46);
              if ( v48 == *v42 )
              {
                if ( v109 == v108 )
                  v41 = (__int64)&v109[HIDWORD(v110)];
                else
                  v41 = (__int64)&v109[(unsigned int)v110];
                goto LABEL_43;
              }
              if ( v109 == v108 )
              {
                v42 = &v109[HIDWORD(v110)];
                v41 = (__int64)v42;
                goto LABEL_43;
              }
              v41 = (unsigned int)v110;
              v42 = &v109[(unsigned int)v110];
            }
            if ( v47 == v42 )
            {
LABEL_46:
              v49 = *v46;
              v46 += 5;
              sub_1F81BC0((__int64)&v94, v49);
              if ( (__int64 *)v68.m128i_i64[0] == v46 )
                break;
            }
            else
            {
LABEL_34:
              v46 += 5;
              if ( (__int64 *)v68.m128i_i64[0] == v46 )
                break;
            }
            v41 = (__int64)v109;
            v42 = v108;
          }
        }
        *((_QWORD *)&v23 + 1) = v35;
        *(_QWORD *)&v23 = &v94;
        v26 = sub_1FD0C90(v23, v41, v45, (__int64)v43, v44, v29, a6, a7);
        v88 = v26;
        v89 = v50;
        if ( (__int64 *)v35 == v26 || !v26 )
          goto LABEL_78;
        if ( *(_DWORD *)(v35 + 60) == *((_DWORD *)v26 + 15) )
          sub_1D444E0((__int64)v94, v35, (__int64)v26);
        else
          sub_1D44A40((__int64)v94, v35, (__int64 *)&v88);
        sub_1F81BC0((__int64)&v94, (__int64)v88);
        for ( i = v88[6]; i; i = *(_QWORD *)(i + 32) )
          sub_1F81BC0((__int64)&v94, *(_QWORD *)(i + 16));
        *((_QWORD *)&v23 + 1) = v35;
        sub_1FC6430((__int64 *)&v94, v35, v51, v52, v53, v54);
        v31 = v75;
        v76[83] = v75;
LABEL_54:
        if ( !(_DWORD)v105 )
          goto LABEL_55;
      }
      v59 = (__int64 **)&v90;
      v88 = 0;
      v89 = 1;
      do
        *v59++ = (__int64 *)-8LL;
      while ( v59 != &v91 );
      *((_QWORD *)&v23 + 1) = v35;
      v91 = (__int64 *)v93;
      v92 = 0x1000000000LL;
      v60 = sub_200CF00(v38, v35, &v88);
      v61 = v91;
      v68.m128i_i8[0] = v60;
      v62 = &v91[(unsigned int)v92];
      if ( v91 != v62 )
      {
        do
        {
          v63 = *v61;
          *((_QWORD *)&v23 + 1) = *v61;
          sub_1F81BC0((__int64)&v94, *v61);
          for ( j = *(_QWORD *)(v63 + 48); j; j = *(_QWORD *)(j + 32) )
          {
            *((_QWORD *)&v23 + 1) = *(_QWORD *)(j + 16);
            sub_1F81BC0((__int64)&v94, *((__int64 *)&v23 + 1));
          }
          ++v61;
        }
        while ( v62 != v61 );
        v62 = v91;
      }
      if ( v68.m128i_i8[0] )
      {
        if ( v62 != (__int64 *)v93 )
          _libc_free((unsigned __int64)v62);
        if ( (v89 & 1) == 0 )
          j___libc_free_0(v90);
        goto LABEL_26;
      }
      if ( v62 != (__int64 *)v93 )
        _libc_free((unsigned __int64)v62);
      if ( (v89 & 1) == 0 )
        j___libc_free_0(v90);
LABEL_78:
      v31 = v75;
      v76[83] = v75;
    }
LABEL_55:
    v56 = v83;
    v57 = v94;
    v58 = v84;
    if ( v83 )
    {
      nullsub_686();
      v72 = v58;
      *((_QWORD *)&v23 + 1) = 0;
      v71 = v56;
      v57[11].m128i_i64[0] = v56;
      v57[11].m128i_i32[2] = v72;
      sub_1D23870();
      v57 = v94;
    }
    else
    {
      v70 = v84;
      v69 = 0;
      v94[11].m128i_i64[0] = 0;
      v57[11].m128i_i32[2] = v70;
    }
    sub_1D2D9C0(v57, *((__int64 *)&v23 + 1), v31, (__int64)v26, v27, (__int64)v28);
    sub_1D189A0((__int64)v78);
  }
  if ( v109 != v108 )
    _libc_free((unsigned __int64)v109);
  j___libc_free_0(v104);
  if ( v100 != v102 )
    _libc_free((unsigned __int64)v100);
}
