// Function: sub_25FBE50
// Address: 0x25fbe50
//
__int64 __fastcall sub_25FBE50(__int64 a1, __int64 *a2, __int64 *a3, unsigned int a4)
{
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 *v9; // rax
  __int64 v10; // r14
  unsigned int v11; // r13d
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rcx
  int v15; // r8d
  _BYTE *v16; // rsi
  int v17; // ecx
  unsigned int v18; // eax
  unsigned int v19; // r8d
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rdx
  __m128i *v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  __m128i *v26; // rdi
  __int64 v27; // r12
  __int64 v28; // r15
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v32; // r12
  unsigned __int8 v33; // dl
  __int64 v34; // r12
  __int64 v35; // r15
  __int64 v36; // rcx
  _BYTE *v37; // r10
  unsigned __int8 v38; // dl
  _QWORD *v39; // rdx
  __int64 v40; // rax
  const char *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rax
  __int64 v48; // r13
  __int64 v49; // r14
  __int64 v50; // r15
  unsigned __int64 v51; // r12
  __int64 v52; // rsi
  char *v53; // r13
  char *v54; // r12
  __int64 v55; // rsi
  unsigned __int64 v56; // r13
  unsigned __int64 v57; // r12
  unsigned __int64 v58; // rdi
  char *v59; // r13
  char *v60; // r12
  __int64 v61; // rsi
  char *v62; // r13
  char *v63; // r12
  __int64 v64; // rsi
  char *v65; // r13
  char *v66; // r12
  __int64 v67; // rsi
  _BYTE *v68; // [rsp+8h] [rbp-298h]
  _BYTE *v69; // [rsp+10h] [rbp-290h]
  __int64 v70; // [rsp+28h] [rbp-278h]
  __int64 v72; // [rsp+30h] [rbp-270h]
  __int64 v73; // [rsp+38h] [rbp-268h]
  __int64 v74; // [rsp+38h] [rbp-268h]
  __int64 v75; // [rsp+38h] [rbp-268h]
  __int64 v76; // [rsp+40h] [rbp-260h] BYREF
  __int64 v77; // [rsp+48h] [rbp-258h]
  __int64 v78; // [rsp+50h] [rbp-250h]
  unsigned int v79; // [rsp+58h] [rbp-248h]
  _QWORD *v80; // [rsp+60h] [rbp-240h] BYREF
  __int64 v81; // [rsp+68h] [rbp-238h]
  _BYTE v82[16]; // [rsp+70h] [rbp-230h] BYREF
  __m128i *v83; // [rsp+80h] [rbp-220h] BYREF
  __int64 v84; // [rsp+88h] [rbp-218h]
  __m128i v85; // [rsp+90h] [rbp-210h] BYREF
  __int64 v86; // [rsp+A0h] [rbp-200h]
  __int64 v87; // [rsp+A8h] [rbp-1F8h]
  _QWORD **v88; // [rsp+B0h] [rbp-1F0h]
  _QWORD v89[4]; // [rsp+C0h] [rbp-1E0h] BYREF
  __int16 v90; // [rsp+E0h] [rbp-1C0h]
  char *v91; // [rsp+F8h] [rbp-1A8h]
  int v92; // [rsp+100h] [rbp-1A0h]
  char v93; // [rsp+108h] [rbp-198h] BYREF
  char *v94; // [rsp+128h] [rbp-178h]
  int v95; // [rsp+130h] [rbp-170h]
  char v96; // [rsp+138h] [rbp-168h] BYREF
  char *v97; // [rsp+158h] [rbp-148h]
  char v98; // [rsp+168h] [rbp-138h] BYREF
  char *v99; // [rsp+188h] [rbp-118h]
  char v100; // [rsp+198h] [rbp-108h] BYREF
  char *v101; // [rsp+1B8h] [rbp-E8h]
  int v102; // [rsp+1C0h] [rbp-E0h]
  char v103; // [rsp+1C8h] [rbp-D8h] BYREF
  __int64 v104; // [rsp+1F0h] [rbp-B0h]
  unsigned int v105; // [rsp+200h] [rbp-A0h]
  unsigned __int64 v106; // [rsp+208h] [rbp-98h]
  unsigned int v107; // [rsp+210h] [rbp-90h]
  char *v108; // [rsp+218h] [rbp-88h] BYREF
  int v109; // [rsp+220h] [rbp-80h]
  char v110; // [rsp+228h] [rbp-78h] BYREF
  __int64 v111; // [rsp+258h] [rbp-48h]
  unsigned int v112; // [rsp+268h] [rbp-38h]

  v6 = sub_BCB120((_QWORD *)*a2);
  v7 = *a3;
  v8 = v6;
  v73 = a3[1];
  if ( *a3 != v73 )
  {
    while ( 1 )
    {
      v9 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v7 + 248LL) + 24LL) + 16LL);
      v10 = *v9;
      if ( *(_BYTE *)(v8 + 8) != 7 || *(_BYTE *)(v10 + 8) == 7 )
      {
        if ( !sub_BCAC40(v8, 1) )
          goto LABEL_5;
        if ( sub_BCAC40(v10, 16) )
          v8 = v10;
        v7 += 8;
        if ( v73 == v7 )
          break;
      }
      else
      {
        v8 = *v9;
LABEL_5:
        v7 += 8;
        if ( v73 == v7 )
          break;
      }
    }
  }
  a3[6] = sub_BCF480((__int64 *)v8, (const void *)a3[3], (a3[4] - a3[3]) >> 3, 0);
  if ( a4 <= 9 )
  {
    v80 = v82;
    sub_2240A50((__int64 *)&v80, 1u, 0);
    v16 = v80;
LABEL_25:
    *v16 = a4 + 48;
    goto LABEL_26;
  }
  if ( a4 <= 0x63 )
  {
    v80 = v82;
    sub_2240A50((__int64 *)&v80, 2u, 0);
    v16 = v80;
  }
  else
  {
    if ( a4 <= 0x3E7 )
    {
      v13 = 3;
      v11 = a4;
    }
    else
    {
      v11 = a4;
      v12 = a4;
      if ( a4 <= 0x270F )
      {
        v13 = 4;
      }
      else
      {
        LODWORD(v13) = 1;
        while ( 1 )
        {
          v14 = v12;
          v15 = v13;
          v13 = (unsigned int)(v13 + 4);
          v12 /= 0x2710u;
          if ( v14 <= 0x1869F )
            break;
          if ( (unsigned int)v12 <= 0x63 )
          {
            v13 = (unsigned int)(v15 + 5);
            v80 = v82;
            goto LABEL_21;
          }
          if ( (unsigned int)v12 <= 0x3E7 )
          {
            v13 = (unsigned int)(v15 + 6);
            break;
          }
          if ( (unsigned int)v12 <= 0x270F )
          {
            v13 = (unsigned int)(v15 + 7);
            break;
          }
        }
      }
    }
    v80 = v82;
LABEL_21:
    sub_2240A50((__int64 *)&v80, v13, 0);
    v16 = v80;
    v17 = v81 - 1;
    while ( 1 )
    {
      v18 = a4 - 100 * (v11 / 0x64);
      v19 = a4;
      a4 = v11 / 0x64;
      v20 = 2 * v18;
      v21 = (unsigned int)(v20 + 1);
      LOBYTE(v20) = a00010203040506[v20];
      v16[v17] = a00010203040506[v21];
      v22 = (unsigned int)(v17 - 1);
      v17 -= 2;
      v16[v22] = v20;
      if ( v19 <= 0x270F )
        break;
      v11 /= 0x64u;
    }
    if ( v19 <= 0x3E7 )
      goto LABEL_25;
  }
  v32 = 2 * a4;
  v16[1] = a00010203040506[(unsigned int)(v32 + 1)];
  *v16 = a00010203040506[v32];
LABEL_26:
  v23 = (__m128i *)sub_2241130((unsigned __int64 *)&v80, 0, 0, "outlined_ir_func_", 0x11u);
  v83 = &v85;
  if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
  {
    v85 = _mm_loadu_si128(v23 + 1);
  }
  else
  {
    v83 = (__m128i *)v23->m128i_i64[0];
    v85.m128i_i64[0] = v23[1].m128i_i64[0];
  }
  v84 = v23->m128i_i64[1];
  v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
  v23->m128i_i64[1] = 0;
  v23[1].m128i_i8[0] = 0;
  v24 = a3[6];
  v90 = 260;
  v89[0] = &v83;
  v25 = sub_B2C660(v24, 7, (__int64)v89, (__int64)a2);
  v26 = v83;
  a3[7] = v25;
  if ( v26 != &v85 )
    j_j___libc_free_0((unsigned __int64)v26);
  if ( v80 != (_QWORD *)v82 )
    j_j___libc_free_0((unsigned __int64)v80);
  if ( *((_BYTE *)a3 + 316) )
    sub_B2D3C0(a3[7], *((_DWORD *)a3 + 78), 74);
  sub_B2CD30(a3[7], 47);
  sub_B2CD30(a3[7], 18);
  v27 = a3[1];
  v28 = *a3;
  if ( *a3 != v27 )
  {
    while ( 1 )
    {
      v29 = sub_B43CB0(*(_QWORD *)(*(_QWORD *)v28 + 240LL));
      if ( v29 )
      {
        v30 = sub_B92180(v29);
        if ( v30 )
          break;
      }
      v28 += 8;
      if ( v27 == v28 )
        return a3[7];
    }
    v33 = *(_BYTE *)(v30 - 16);
    v34 = a3[7];
    v35 = v30 - 16;
    if ( (v33 & 2) != 0 )
      v36 = *(_QWORD *)(v30 - 32);
    else
      v36 = v35 - 8LL * ((v33 >> 2) & 0xF);
    v69 = (_BYTE *)v30;
    sub_AE0470((__int64)v89, a2, 1, *(_QWORD *)(v36 + 40));
    v37 = v69;
    if ( *v69 != 16 )
    {
      v38 = *(v69 - 16);
      if ( (v38 & 2) != 0 )
        v39 = (_QWORD *)*((_QWORD *)v69 - 4);
      else
        v39 = (_QWORD *)(v35 - 8LL * ((v38 >> 2) & 0xF));
      v37 = (_BYTE *)*v39;
    }
    v68 = v37;
    v80 = v82;
    v87 = 0x100000000LL;
    v76 = 0;
    v77 = 0;
    v83 = (__m128i *)&unk_49DD210;
    v78 = 0;
    v79 = 0;
    v81 = 0;
    v82[0] = 0;
    v84 = 0;
    v85 = 0u;
    v86 = 0;
    v88 = &v80;
    sub_CB5980((__int64)&v83, 0, 0, 0);
    sub_E409B0((__int64)&v76, (__int64)&v83, v34);
    v40 = sub_ADD430((__int64)v89, 0, 0);
    v74 = sub_ADCD40((__int64)v89, v40, 0, 0);
    v70 = (__int64)v80;
    v72 = v81;
    v41 = sub_BD5D20(v34);
    v75 = sub_ADE3D0(
            (__int64)v89,
            v68,
            (__int64)v41,
            v42,
            v70,
            v72,
            (__int64)v68,
            0,
            v74,
            0,
            64,
            24,
            0,
            0,
            0,
            0,
            (__int64)byte_3F871B3,
            0);
    sub_ADC590((__int64)v89, v75);
    sub_B994C0(v34, v75);
    sub_ADCDB0((__int64)v89, v75, v43, v44, v45, v46);
    v83 = (__m128i *)&unk_49DD210;
    sub_CB5840((__int64)&v83);
    sub_2240A30((unsigned __int64 *)&v80);
    sub_C7D6A0(v77, 16LL * v79, 8);
    v47 = v112;
    if ( v112 )
    {
      v48 = v111;
      v49 = v111 + 56LL * v112;
      do
      {
        if ( *(_QWORD *)v48 != -4096 && *(_QWORD *)v48 != -8192 )
        {
          v50 = *(_QWORD *)(v48 + 8);
          v51 = v50 + 8LL * *(unsigned int *)(v48 + 16);
          if ( v50 != v51 )
          {
            do
            {
              v52 = *(_QWORD *)(v51 - 8);
              v51 -= 8LL;
              if ( v52 )
                sub_B91220(v51, v52);
            }
            while ( v50 != v51 );
            v51 = *(_QWORD *)(v48 + 8);
          }
          if ( v51 != v48 + 24 )
            _libc_free(v51);
        }
        v48 += 56;
      }
      while ( v49 != v48 );
      v47 = v112;
    }
    sub_C7D6A0(v111, 56 * v47, 8);
    v53 = v108;
    v54 = &v108[8 * v109];
    if ( v108 != v54 )
    {
      do
      {
        v55 = *((_QWORD *)v54 - 1);
        v54 -= 8;
        if ( v55 )
          sub_B91220((__int64)v54, v55);
      }
      while ( v53 != v54 );
      v54 = v108;
    }
    if ( v54 != &v110 )
      _libc_free((unsigned __int64)v54);
    v56 = v106;
    v57 = v106 + 56LL * v107;
    if ( v106 != v57 )
    {
      do
      {
        v57 -= 56LL;
        v58 = *(_QWORD *)(v57 + 40);
        if ( v58 != v57 + 56 )
          _libc_free(v58);
        sub_C7D6A0(*(_QWORD *)(v57 + 16), 8LL * *(unsigned int *)(v57 + 32), 8);
      }
      while ( v56 != v57 );
      v57 = v106;
    }
    if ( (char **)v57 != &v108 )
      _libc_free(v57);
    sub_C7D6A0(v104, 16LL * v105, 8);
    v59 = v101;
    v60 = &v101[8 * v102];
    if ( v101 != v60 )
    {
      do
      {
        v61 = *((_QWORD *)v60 - 1);
        v60 -= 8;
        if ( v61 )
          sub_B91220((__int64)v60, v61);
      }
      while ( v59 != v60 );
      v60 = v101;
    }
    if ( v60 != &v103 )
      _libc_free((unsigned __int64)v60);
    if ( v99 != &v100 )
      _libc_free((unsigned __int64)v99);
    if ( v97 != &v98 )
      _libc_free((unsigned __int64)v97);
    v62 = v94;
    v63 = &v94[8 * v95];
    if ( v94 != v63 )
    {
      do
      {
        v64 = *((_QWORD *)v63 - 1);
        v63 -= 8;
        if ( v64 )
          sub_B91220((__int64)v63, v64);
      }
      while ( v62 != v63 );
      v63 = v94;
    }
    if ( v63 != &v96 )
      _libc_free((unsigned __int64)v63);
    v65 = v91;
    v66 = &v91[8 * v92];
    if ( v91 != v66 )
    {
      do
      {
        v67 = *((_QWORD *)v66 - 1);
        v66 -= 8;
        if ( v67 )
          sub_B91220((__int64)v66, v67);
      }
      while ( v65 != v66 );
      v66 = v91;
    }
    if ( v66 != &v93 )
      _libc_free((unsigned __int64)v66);
  }
  return a3[7];
}
