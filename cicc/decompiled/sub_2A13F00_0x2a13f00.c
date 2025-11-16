// Function: sub_2A13F00
// Address: 0x2a13f00
//
void __fastcall sub_2A13F00(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int64 a8)
{
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r15
  unsigned __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // r8
  int v22; // edx
  __int64 v23; // r10
  int v24; // edx
  unsigned int v25; // ecx
  __int64 *v26; // rax
  __int64 v27; // r11
  _QWORD *v28; // rax
  unsigned int v29; // edi
  __int64 *v30; // rcx
  __int64 v31; // r11
  _QWORD *v32; // rdx
  unsigned __int64 v33; // rbx
  char **v34; // r12
  __int64 v35; // rax
  _BYTE *v36; // rax
  __int64 v37; // rdi
  __int64 v38; // r10
  __int64 v39; // rdi
  unsigned __int8 v40; // al
  __int64 v41; // rdx
  unsigned __int8 **v42; // rax
  unsigned __int8 *v43; // r13
  unsigned __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rdx
  unsigned __int64 v48; // rax
  __int64 v49; // rcx
  __int64 v50; // rcx
  char v51; // si
  char v52; // si
  int v53; // eax
  _QWORD *v54; // rdi
  int v55; // eax
  _QWORD *v56; // rdi
  _BYTE *v57; // rax
  _BYTE *v58; // rax
  unsigned __int64 *v59; // r13
  unsigned __int64 *v60; // rdi
  __int32 v61; // eax
  int v62; // eax
  int v63; // r9d
  int v64; // ecx
  int v65; // r9d
  unsigned __int64 *v66; // r12
  unsigned __int64 *v67; // rdi
  int v68; // r13d
  __int64 v70; // r9
  unsigned __int32 i; // eax
  __int64 v72; // rcx
  __int64 v73; // rax
  _QWORD *v74; // rdi
  __int64 v75; // rax
  char *v76; // r15
  __int64 *v77; // rax
  __int64 *v78; // rax
  unsigned __int64 v79; // r12
  __int64 v80; // r14
  _QWORD *v81; // r12
  __int64 v82; // rax
  __int64 *v83; // [rsp+10h] [rbp-4F0h]
  __int64 *v85; // [rsp+28h] [rbp-4D8h]
  unsigned __int8 *v86; // [rsp+30h] [rbp-4D0h]
  __int64 v87; // [rsp+30h] [rbp-4D0h]
  __int64 v88; // [rsp+38h] [rbp-4C8h]
  __int64 v89; // [rsp+38h] [rbp-4C8h]
  __int64 v90; // [rsp+38h] [rbp-4C8h]
  __int32 v91; // [rsp+38h] [rbp-4C8h]
  __int64 v93; // [rsp+48h] [rbp-4B8h]
  __int64 v95; // [rsp+58h] [rbp-4A8h]
  unsigned __int64 v96; // [rsp+68h] [rbp-498h] BYREF
  unsigned __int64 v97[4]; // [rsp+70h] [rbp-490h] BYREF
  __m128i v98; // [rsp+90h] [rbp-470h] BYREF
  _QWORD v99[6]; // [rsp+A0h] [rbp-460h] BYREF
  __int16 v100; // [rsp+D0h] [rbp-430h]
  unsigned __int64 v101; // [rsp+220h] [rbp-2E0h] BYREF
  unsigned __int64 v102; // [rsp+228h] [rbp-2D8h]
  char *v103; // [rsp+230h] [rbp-2D0h] BYREF
  __int64 v104; // [rsp+238h] [rbp-2C8h]
  __int64 v105; // [rsp+240h] [rbp-2C0h] BYREF
  unsigned int v106; // [rsp+248h] [rbp-2B8h]
  _QWORD v107[2]; // [rsp+380h] [rbp-180h] BYREF
  char v108; // [rsp+390h] [rbp-170h]
  _BYTE *v109; // [rsp+398h] [rbp-168h]
  __int64 v110; // [rsp+3A0h] [rbp-160h]
  _BYTE v111[128]; // [rsp+3A8h] [rbp-158h] BYREF
  __int16 v112; // [rsp+428h] [rbp-D8h]
  _QWORD v113[2]; // [rsp+430h] [rbp-D0h] BYREF
  __int64 v114; // [rsp+440h] [rbp-C0h]
  __int64 v115; // [rsp+448h] [rbp-B8h] BYREF
  unsigned int v116; // [rsp+450h] [rbp-B0h]
  char v117; // [rsp+4C8h] [rbp-38h] BYREF

  if ( a4 && (_BYTE)a2 )
  {
    v98.m128i_i64[1] = 0x1000000000LL;
    a2 = a4;
    v98.m128i_i64[0] = (__int64)v99;
    sub_2A79930(a1, a4, a5, a3, a7, &v98);
    for ( i = v98.m128i_u32[2]; v98.m128i_i32[2]; i = v98.m128i_u32[2] )
    {
      v72 = v98.m128i_i64[0];
      v101 = 6;
      v102 = 0;
      a2 = v98.m128i_i64[0] + 24LL * i - 24;
      v103 = *(char **)(a2 + 16);
      if ( v103 != 0 && v103 + 4096 != 0 && v103 != (char *)-8192LL )
      {
        a2 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
        sub_BD6050(&v101, a2);
        v72 = v98.m128i_i64[0];
        i = v98.m128i_u32[2];
      }
      v73 = i - 1;
      v98.m128i_i32[2] = v73;
      v74 = (_QWORD *)(v72 + 24 * v73);
      v75 = v74[2];
      if ( v75 != 0 && v75 != -4096 && v75 != -8192 )
        sub_BD60C0(v74);
      v76 = v103;
      if ( v103 )
      {
        if ( v103 != (char *)-4096LL && v103 != (char *)-8192LL )
          sub_BD60C0(&v101);
        if ( (unsigned __int8)*v76 > 0x1Cu )
        {
          a2 = 0;
          v103 = 0;
          sub_F5CAB0(v76, 0, 0, (__int64)&v101);
          if ( v103 )
          {
            a2 = (__int64)&v101;
            ((void (__fastcall *)(unsigned __int64 *, unsigned __int64 *, __int64))v103)(&v101, &v101, 3);
          }
        }
      }
    }
    if ( a8 )
    {
      v96 = 0;
      v103 = 0;
      v104 = 1;
      v101 = a8;
      v102 = a8;
      v77 = &v105;
      do
      {
        *v77 = -4;
        v77 += 5;
        *(v77 - 4) = -3;
        *(v77 - 3) = -4;
        *(v77 - 2) = -3;
      }
      while ( v77 != v107 );
      v110 = 0x400000000LL;
      v107[0] = v113;
      v107[1] = 0;
      v108 = 0;
      v109 = v111;
      v112 = 256;
      v113[1] = 0;
      v114 = 1;
      v113[0] = &unk_49DDBE8;
      v78 = &v115;
      do
      {
        *v78 = -4096;
        v78 += 2;
      }
      while ( v78 != (__int64 *)&v117 );
      a2 = a5;
      v97[1] = a8;
      v97[3] = (unsigned __int64)&v96;
      v97[0] = a1;
      v97[2] = a5;
      sub_2A134F0(a1, a5, a4, a3, (__int64)&v101, v70, (__int64 (__fastcall *)(__int64))sub_2A109A0, (__int64)v97);
      v113[0] = &unk_49DDBE8;
      if ( (v114 & 1) == 0 )
      {
        a2 = 16LL * v116;
        sub_C7D6A0(v115, a2, 8);
      }
      nullsub_184();
      if ( v109 != v111 )
        _libc_free((unsigned __int64)v109);
      if ( (v104 & 1) == 0 )
      {
        a2 = 40LL * v106;
        sub_C7D6A0(v105, a2, 8);
      }
      v79 = v96;
      if ( v96 )
      {
        sub_103C970(v96);
        a2 = 360;
        j_j___libc_free_0(v79);
      }
    }
    v80 = v98.m128i_i64[0];
    v81 = (_QWORD *)(v98.m128i_i64[0] + 24LL * v98.m128i_u32[2]);
    if ( (_QWORD *)v98.m128i_i64[0] != v81 )
    {
      do
      {
        v82 = *(v81 - 1);
        v81 -= 3;
        if ( v82 != 0 && v82 != -4096 && v82 != -8192 )
          sub_BD60C0(v81);
      }
      while ( (_QWORD *)v80 != v81 );
      v81 = (_QWORD *)v98.m128i_i64[0];
    }
    if ( v81 != v99 )
      _libc_free((unsigned __int64)v81);
  }
  v93 = sub_AA4E30(**(_QWORD **)(a1 + 32));
  v101 = (unsigned __int64)&v103;
  v102 = 0x1000000000LL;
  v83 = *(__int64 **)(a1 + 40);
  if ( *(__int64 **)(a1 + 32) != v83 )
  {
    v85 = *(__int64 **)(a1 + 32);
    do
    {
      v9 = *v85;
      if ( sub_B92180(*(_QWORD *)(*v85 + 72)) )
        sub_F3F2F0(v9, a2);
      v14 = v9 + 48;
      v15 = *(_QWORD *)(v9 + 56);
      v95 = v14;
      if ( v14 != v15 )
      {
        while ( 1 )
        {
          v16 = v15;
          v15 = *(_QWORD *)(v15 + 8);
          v17 = v16 - 24;
          v98 = (__m128i)(unsigned __int64)v93;
          v99[0] = 0;
          v99[1] = a5;
          memset(&v99[3], 0, 24);
          v99[2] = a6;
          v100 = 257;
          v18 = sub_1020E10(v16 - 24, &v98, v10, v11, v12, v13);
          v19 = v18;
          if ( !v18 )
            goto LABEL_19;
          if ( *(_BYTE *)v18 <= 0x1Cu )
            goto LABEL_18;
          v20 = *(_QWORD *)(v18 + 40);
          v21 = *(_QWORD *)(v16 + 16);
          if ( v20 == v21 )
            goto LABEL_18;
          v22 = *(_DWORD *)(a3 + 24);
          v23 = *(_QWORD *)(a3 + 8);
          if ( !v22 )
            goto LABEL_18;
          v24 = v22 - 1;
          v25 = v24 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v26 = (__int64 *)(v23 + 16LL * v25);
          v27 = *v26;
          if ( *v26 != v20 )
            break;
LABEL_13:
          v28 = (_QWORD *)v26[1];
          if ( !v28 )
            goto LABEL_18;
          v29 = v24 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v30 = (__int64 *)(v23 + 16LL * v29);
          v31 = *v30;
          if ( v21 == *v30 )
          {
LABEL_15:
            v32 = (_QWORD *)v30[1];
            if ( v32 == v28 )
              goto LABEL_18;
            while ( v32 )
            {
              v32 = (_QWORD *)*v32;
              if ( v32 == v28 )
                goto LABEL_18;
            }
          }
          else
          {
            v64 = 1;
            while ( v31 != -4096 )
            {
              v65 = v64 + 1;
              v29 = v24 & (v64 + v29);
              v30 = (__int64 *)(v23 + 16LL * v29);
              v31 = *v30;
              if ( v21 == *v30 )
                goto LABEL_15;
              v64 = v65;
            }
          }
LABEL_19:
          if ( sub_F50EE0((unsigned __int8 *)(v16 - 24), 0) )
          {
            v53 = v102;
            if ( HIDWORD(v102) <= (unsigned int)v102 )
            {
              v59 = (unsigned __int64 *)sub_C8D7D0(
                                          (__int64)&v101,
                                          (__int64)&v103,
                                          0,
                                          0x18u,
                                          (unsigned __int64 *)&v98,
                                          v13);
              v60 = &v59[3 * (unsigned int)v102];
              if ( v60 )
              {
                *v60 = 6;
                v60[1] = 0;
                v60[2] = v17;
                if ( v16 != -8168 && v16 != -4072 )
                  sub_BD73F0((__int64)v60);
              }
              sub_F17F80((__int64)&v101, v59);
              v61 = v98.m128i_i32[0];
              if ( (char **)v101 != &v103 )
              {
                v91 = v98.m128i_i32[0];
                _libc_free(v101);
                v61 = v91;
              }
              LODWORD(v102) = v102 + 1;
              v101 = (unsigned __int64)v59;
              HIDWORD(v102) = v61;
            }
            else
            {
              v11 = 3LL * (unsigned int)v102;
              v10 = v101;
              v54 = (_QWORD *)(v101 + 24LL * (unsigned int)v102);
              if ( v54 )
              {
                *v54 = 6;
                v54[1] = 0;
                v54[2] = v17;
                if ( v16 != -4072 && v16 != -8168 )
                  sub_BD73F0((__int64)v54);
                v53 = v102;
              }
              LODWORD(v102) = v53 + 1;
            }
          }
          if ( *(_BYTE *)(v16 - 24) != 42 )
            goto LABEL_21;
          v36 = *(_BYTE **)(v16 - 88);
          if ( *v36 != 42 )
            goto LABEL_21;
          v12 = *((_QWORD *)v36 - 8);
          if ( !v12 )
            goto LABEL_21;
          v37 = *((_QWORD *)v36 - 4);
          v38 = v37 + 24;
          if ( *(_BYTE *)v37 == 17 )
          {
            v39 = *(_QWORD *)(v16 - 56);
            v40 = *(_BYTE *)v39;
            v41 = v39 + 24;
            if ( *(_BYTE *)v39 == 17 )
              goto LABEL_38;
          }
          else
          {
            v89 = *((_QWORD *)v36 - 8);
            v10 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v37 + 8) + 8LL) - 17;
            if ( (unsigned int)v10 > 1 )
              goto LABEL_21;
            if ( *(_BYTE *)v37 > 0x15u )
              goto LABEL_21;
            v57 = sub_AD7630(v37, 0, v10);
            if ( !v57 || *v57 != 17 )
              goto LABEL_21;
            v39 = *(_QWORD *)(v16 - 56);
            v38 = (__int64)(v57 + 24);
            v12 = v89;
            v40 = *(_BYTE *)v39;
            v41 = v39 + 24;
            if ( *(_BYTE *)v39 == 17 )
              goto LABEL_38;
          }
          v87 = v38;
          v90 = v12;
          v10 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v39 + 8) + 8LL) - 17;
          if ( (unsigned int)v10 <= 1 && v40 <= 0x15u )
          {
            v58 = sub_AD7630(v39, 0, v10);
            if ( v58 )
            {
              if ( *v58 == 17 )
              {
                v38 = v87;
                v12 = v90;
                v41 = (__int64)(v58 + 24);
LABEL_38:
                if ( (*(_BYTE *)(v16 - 17) & 0x40) != 0 )
                {
                  v42 = *(unsigned __int8 ***)(v16 - 32);
                  v43 = *v42;
                  v86 = *v42;
                  if ( **v42 > 0x1Cu )
                  {
LABEL_40:
                    v88 = v12;
                    sub_C45F70((__int64)&v98, v38, v41, &v96);
                    if ( (*(_BYTE *)(v16 - 17) & 0x40) != 0 )
                      v44 = *(_QWORD *)(v16 - 32);
                    else
                      v44 = v17 - 32LL * (*(_DWORD *)(v16 - 20) & 0x7FFFFFF);
                    if ( *(_QWORD *)v44 )
                    {
                      v45 = *(_QWORD *)(v44 + 8);
                      **(_QWORD **)(v44 + 16) = v45;
                      if ( v45 )
                        *(_QWORD *)(v45 + 16) = *(_QWORD *)(v44 + 16);
                    }
                    *(_QWORD *)v44 = v88;
                    v46 = *(_QWORD *)(v88 + 16);
                    *(_QWORD *)(v44 + 8) = v46;
                    if ( v46 )
                      *(_QWORD *)(v46 + 16) = v44 + 8;
                    *(_QWORD *)(v44 + 16) = v88 + 16;
                    *(_QWORD *)(v88 + 16) = v44;
                    v47 = sub_AD8D80(*(_QWORD *)(v16 - 16), (__int64)&v98);
                    if ( (*(_BYTE *)(v16 - 17) & 0x40) != 0 )
                      v48 = *(_QWORD *)(v16 - 32);
                    else
                      v48 = v17 - 32LL * (*(_DWORD *)(v16 - 20) & 0x7FFFFFF);
                    if ( *(_QWORD *)(v48 + 32) )
                    {
                      v49 = *(_QWORD *)(v48 + 40);
                      **(_QWORD **)(v48 + 48) = v49;
                      if ( v49 )
                        *(_QWORD *)(v49 + 16) = *(_QWORD *)(v48 + 48);
                    }
                    *(_QWORD *)(v48 + 32) = v47;
                    if ( v47 )
                    {
                      v50 = *(_QWORD *)(v47 + 16);
                      *(_QWORD *)(v48 + 40) = v50;
                      if ( v50 )
                        *(_QWORD *)(v50 + 16) = v48 + 40;
                      *(_QWORD *)(v48 + 48) = v47 + 16;
                      *(_QWORD *)(v47 + 16) = v48 + 32;
                    }
                    v51 = 0;
                    if ( sub_B448F0(v16 - 24) )
                      v51 = (v86[1] & 2) != 0;
                    sub_B447F0((unsigned __int8 *)(v16 - 24), v51);
                    v52 = 0;
                    if ( sub_B44900(v16 - 24) && (v86[1] & 4) != 0 )
                      v52 = v96 ^ 1;
                    sub_B44850((unsigned __int8 *)(v16 - 24), v52);
                    if ( v43 && sub_F50EE0(v43, 0) )
                    {
                      v55 = v102;
                      if ( HIDWORD(v102) <= (unsigned int)v102 )
                      {
                        v66 = (unsigned __int64 *)sub_C8D7D0((__int64)&v101, (__int64)&v103, 0, 0x18u, v97, v13);
                        v67 = &v66[3 * (unsigned int)v102];
                        if ( v67 )
                        {
                          *v67 = 6;
                          v67[1] = 0;
                          v67[2] = (unsigned __int64)v43;
                          if ( v43 != (unsigned __int8 *)-8192LL && v43 != (unsigned __int8 *)-4096LL )
                            sub_BD73F0((__int64)v67);
                        }
                        sub_F17F80((__int64)&v101, v66);
                        v68 = v97[0];
                        if ( (char **)v101 != &v103 )
                          _libc_free(v101);
                        LODWORD(v102) = v102 + 1;
                        v101 = (unsigned __int64)v66;
                        HIDWORD(v102) = v68;
                      }
                      else
                      {
                        v11 = 3LL * (unsigned int)v102;
                        v10 = v101;
                        v56 = (_QWORD *)(v101 + 24LL * (unsigned int)v102);
                        if ( v56 )
                        {
                          *v56 = 6;
                          v56[1] = 0;
                          v56[2] = v43;
                          if ( v43 != (unsigned __int8 *)-8192LL && v43 != (unsigned __int8 *)-4096LL )
                            sub_BD73F0((__int64)v56);
                          v55 = v102;
                        }
                        LODWORD(v102) = v55 + 1;
                      }
                    }
                    if ( v98.m128i_i32[2] > 0x40u && v98.m128i_i64[0] )
                      j_j___libc_free_0_0(v98.m128i_u64[0]);
                    goto LABEL_21;
                  }
                }
                else
                {
                  v43 = *(unsigned __int8 **)(v17 - 32LL * (*(_DWORD *)(v16 - 20) & 0x7FFFFFF));
                  v86 = v43;
                  if ( *v43 > 0x1Cu )
                    goto LABEL_40;
                }
                v43 = 0;
                goto LABEL_40;
              }
            }
          }
LABEL_21:
          if ( v95 == v15 )
            goto LABEL_22;
        }
        v62 = 1;
        while ( v27 != -4096 )
        {
          v63 = v62 + 1;
          v25 = v24 & (v62 + v25);
          v26 = (__int64 *)(v23 + 16LL * v25);
          v27 = *v26;
          if ( v20 == *v26 )
            goto LABEL_13;
          v62 = v63;
        }
LABEL_18:
        sub_BD84D0(v16 - 24, v19);
        goto LABEL_19;
      }
LABEL_22:
      a2 = 0;
      v99[0] = 0;
      sub_F5C330((__int64)&v101, 0, 0, (__int64)&v98);
      if ( v99[0] )
      {
        a2 = (__int64)&v98;
        ((void (__fastcall *)(__m128i *, __m128i *, __int64))v99[0])(&v98, &v98, 3);
      }
      ++v85;
    }
    while ( v83 != v85 );
    v33 = v101;
    v34 = (char **)(v101 + 24LL * (unsigned int)v102);
    if ( (char **)v101 != v34 )
    {
      do
      {
        v35 = (__int64)*(v34 - 1);
        v34 -= 3;
        if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
          sub_BD60C0(v34);
      }
      while ( (char **)v33 != v34 );
      v34 = (char **)v101;
    }
    if ( v34 != &v103 )
      _libc_free((unsigned __int64)v34);
  }
}
