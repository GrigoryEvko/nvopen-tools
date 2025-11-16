// Function: sub_38AFC80
// Address: 0x38afc80
//
__int64 __fastcall sub_38AFC80(__int64 a1, __int64 *a2, __int64 *a3, double a4, __m128i a5, double a6)
{
  __int64 v6; // rbx
  __int16 *v7; // r13
  unsigned int v8; // r12d
  unsigned __int64 *v9; // rbx
  unsigned __int64 *v10; // r15
  unsigned __int64 v11; // rdi
  unsigned __int64 v13; // rax
  __int64 *v14; // rdi
  __int64 v15; // rax
  int v16; // r9d
  __int64 v17; // rax
  __int64 *v18; // r12
  __int64 *v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rsi
  _BYTE *v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rdx
  _BYTE *v25; // r12
  __int64 v26; // r13
  __int64 v27; // rsi
  __int64 v28; // rbx
  __int64 v29; // r12
  unsigned __int64 v30; // rax
  const char *v31; // rax
  unsigned int v32; // eax
  __int64 v33; // rax
  int v34; // edi
  __int64 *v35; // r15
  __int64 *i; // rax
  __int64 v37; // rdx
  _QWORD *v38; // rax
  unsigned int v39; // r8d
  __int64 v40; // r10
  __int64 v41; // r15
  unsigned __int64 *v42; // rax
  int v43; // r8d
  __int64 *v44; // [rsp+0h] [rbp-5A0h]
  __int64 v45; // [rsp+8h] [rbp-598h]
  __int64 v46; // [rsp+8h] [rbp-598h]
  unsigned __int64 v47; // [rsp+28h] [rbp-578h]
  __int64 *v48; // [rsp+28h] [rbp-578h]
  __int64 v49; // [rsp+28h] [rbp-578h]
  __int64 v50; // [rsp+30h] [rbp-570h]
  __int64 *v51; // [rsp+38h] [rbp-568h]
  __int64 v52; // [rsp+38h] [rbp-568h]
  __int64 v53; // [rsp+38h] [rbp-568h]
  unsigned __int64 v55; // [rsp+48h] [rbp-558h]
  char *v56; // [rsp+48h] [rbp-558h]
  __int64 v57; // [rsp+48h] [rbp-558h]
  unsigned __int64 v59; // [rsp+50h] [rbp-550h]
  unsigned int v60; // [rsp+50h] [rbp-550h]
  __int64 *v61; // [rsp+50h] [rbp-550h]
  __int64 v62; // [rsp+60h] [rbp-540h]
  __int64 v63; // [rsp+60h] [rbp-540h]
  __int64 v64; // [rsp+60h] [rbp-540h]
  __int64 v65; // [rsp+60h] [rbp-540h]
  __int64 v66; // [rsp+70h] [rbp-530h]
  int v67; // [rsp+94h] [rbp-50Ch] BYREF
  __int64 v68; // [rsp+98h] [rbp-508h] BYREF
  __int64 *v69; // [rsp+A0h] [rbp-500h] BYREF
  __int64 v70; // [rsp+A8h] [rbp-4F8h] BYREF
  __int64 v71; // [rsp+B0h] [rbp-4F0h] BYREF
  __int64 v72; // [rsp+B8h] [rbp-4E8h] BYREF
  char *v73[4]; // [rsp+C0h] [rbp-4E0h] BYREF
  __m128i *v74; // [rsp+E0h] [rbp-4C0h] BYREF
  __int16 v75; // [rsp+F0h] [rbp-4B0h]
  unsigned __int64 v76[4]; // [rsp+100h] [rbp-4A0h] BYREF
  __m128i v77[2]; // [rsp+120h] [rbp-480h] BYREF
  __m128i v78; // [rsp+140h] [rbp-460h] BYREF
  __int16 v79; // [rsp+150h] [rbp-450h]
  __int64 *v80; // [rsp+160h] [rbp-440h] BYREF
  __int64 v81; // [rsp+168h] [rbp-438h]
  _QWORD v82[8]; // [rsp+170h] [rbp-430h] BYREF
  char *v83; // [rsp+1B0h] [rbp-3F0h] BYREF
  __int64 v84; // [rsp+1B8h] [rbp-3E8h]
  char v85; // [rsp+1C0h] [rbp-3E0h] BYREF
  char v86; // [rsp+1C1h] [rbp-3DFh]
  __m128i v87; // [rsp+200h] [rbp-3A0h] BYREF
  int v88; // [rsp+210h] [rbp-390h] BYREF
  _QWORD *v89; // [rsp+218h] [rbp-388h]
  int *v90; // [rsp+220h] [rbp-380h]
  int *v91; // [rsp+228h] [rbp-378h]
  __int64 v92; // [rsp+230h] [rbp-370h]
  __int64 v93; // [rsp+238h] [rbp-368h]
  __int64 v94; // [rsp+240h] [rbp-360h]
  __int64 v95; // [rsp+248h] [rbp-358h]
  __int64 v96; // [rsp+250h] [rbp-350h]
  __int64 v97; // [rsp+258h] [rbp-348h]
  __m128i v98; // [rsp+260h] [rbp-340h] BYREF
  int v99; // [rsp+270h] [rbp-330h] BYREF
  _QWORD *v100; // [rsp+278h] [rbp-328h]
  int *v101; // [rsp+280h] [rbp-320h]
  int *v102; // [rsp+288h] [rbp-318h]
  __int64 v103; // [rsp+290h] [rbp-310h]
  __int64 v104; // [rsp+298h] [rbp-308h]
  __int64 v105; // [rsp+2A0h] [rbp-300h]
  __int64 v106; // [rsp+2A8h] [rbp-2F8h]
  __int64 v107; // [rsp+2B0h] [rbp-2F0h]
  __int64 v108; // [rsp+2B8h] [rbp-2E8h]
  __int64 *v109; // [rsp+2C0h] [rbp-2E0h] BYREF
  __int64 v110; // [rsp+2C8h] [rbp-2D8h]
  _BYTE v111[112]; // [rsp+2D0h] [rbp-2D0h] BYREF
  unsigned int v112; // [rsp+340h] [rbp-260h] BYREF
  __int64 v113; // [rsp+348h] [rbp-258h]
  __int64 *v114; // [rsp+358h] [rbp-248h]
  _QWORD *v115; // [rsp+360h] [rbp-240h]
  __int64 v116; // [rsp+368h] [rbp-238h]
  _BYTE v117[16]; // [rsp+370h] [rbp-230h] BYREF
  _QWORD *v118; // [rsp+380h] [rbp-220h]
  __int64 v119; // [rsp+388h] [rbp-218h]
  _BYTE v120[16]; // [rsp+390h] [rbp-210h] BYREF
  unsigned __int64 v121; // [rsp+3A0h] [rbp-200h]
  unsigned int v122; // [rsp+3A8h] [rbp-1F8h]
  char v123; // [rsp+3ACh] [rbp-1F4h]
  void *v124; // [rsp+3B8h] [rbp-1E8h] BYREF
  __int64 v125; // [rsp+3C0h] [rbp-1E0h]
  unsigned __int64 v126; // [rsp+3D8h] [rbp-1C8h]
  _BYTE *v127; // [rsp+3E0h] [rbp-1C0h] BYREF
  __int64 v128; // [rsp+3E8h] [rbp-1B8h]
  _BYTE v129[432]; // [rsp+3F0h] [rbp-1B0h] BYREF

  v6 = a1;
  v55 = *(_QWORD *)(a1 + 56);
  v90 = &v88;
  v91 = &v88;
  v101 = &v99;
  v102 = &v99;
  v87.m128i_i64[0] = 0;
  v88 = 0;
  v89 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98.m128i_i64[0] = 0;
  v99 = 0;
  v100 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v115 = v117;
  v108 = 0;
  memset(v73, 0, 24);
  v68 = 0;
  v69 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v116 = 0;
  v117[0] = 0;
  v118 = v120;
  v119 = 0;
  v120[0] = 0;
  v122 = 1;
  v121 = 0;
  v123 = 0;
  v7 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)&v127, 0.0);
  sub_169E320(&v124, (__int64 *)&v127, v7);
  sub_1698460((__int64)&v127);
  v127 = v129;
  v128 = 0x1000000000LL;
  v126 = 0;
  v109 = (__int64 *)v111;
  v110 = 0x200000000LL;
  if ( (unsigned __int8)sub_388C2C0(a1, &v67) )
    goto LABEL_2;
  if ( (unsigned __int8)sub_388C990(a1, &v87) )
    goto LABEL_2;
  v13 = *(_QWORD *)(a1 + 56);
  v86 = 1;
  v47 = v13;
  v83 = "expected type";
  v85 = 3;
  if ( (unsigned __int8)sub_3891B00(a1, (__int64 *)&v69, (__int64)&v83, 1) )
    goto LABEL_2;
  if ( (unsigned __int8)sub_389C540(a1, (__int64)&v112, 0.0, *(double *)a5.m128i_i64, a6) )
    goto LABEL_2;
  if ( (unsigned __int8)sub_38AF780(a1, (__int64)&v127, a3, 0, 0, (__m128)0LL, *(double *)a5.m128i_i64, a6) )
    goto LABEL_2;
  if ( (unsigned __int8)sub_388FCA0(a1, &v98, (__int64)v73, 0, &v68) )
    goto LABEL_2;
  if ( (unsigned __int8)sub_38A1290(a1, (__int64)&v109, a3, (__m128i)0LL, a5, a6) )
    goto LABEL_2;
  if ( (unsigned __int8)sub_388AF10(a1, 53, "expected 'to' in invoke") )
    goto LABEL_2;
  v83 = 0;
  if ( (unsigned __int8)sub_38AB2F0(a1, &v70, (unsigned __int64 *)&v83, a3, 0.0, *(double *)a5.m128i_i64, a6) )
    goto LABEL_2;
  if ( (unsigned __int8)sub_388AF10(a1, 63, "expected 'unwind' in invoke") )
    goto LABEL_2;
  v83 = 0;
  if ( (unsigned __int8)sub_38AB2F0(a1, &v71, (unsigned __int64 *)&v83, a3, 0.0, *(double *)a5.m128i_i64, a6) )
    goto LABEL_2;
  v14 = v69;
  v51 = v69;
  if ( *((_BYTE *)v69 + 8) != 12 )
  {
    v80 = 0;
    v81 = 0;
    v82[0] = 0;
    if ( (_DWORD)v128 )
    {
      v29 = 0;
      v52 = 24LL * (unsigned int)v128;
      do
      {
        v30 = **(_QWORD **)&v127[v29 + 8];
        v83 = (char *)v30;
        if ( v81 == v82[0] )
        {
          sub_1278040((__int64)&v80, (_BYTE *)v81, &v83);
        }
        else
        {
          if ( v81 )
            *(_QWORD *)v81 = v30;
          v81 += 8;
        }
        v29 += 24;
      }
      while ( v52 != v29 );
      v14 = v69;
    }
    if ( !(unsigned __int8)sub_1643460((__int64)v14) )
    {
      v86 = 1;
      v83 = "Invalid result type for LLVM function";
      v85 = 3;
      v8 = sub_38814C0(v6 + 8, v47, (__int64)&v83);
      sub_388FA10((unsigned __int64 *)&v80);
      goto LABEL_3;
    }
    v51 = (__int64 *)sub_1644EA0(v69, v80, (v81 - (__int64)v80) >> 3, 0);
    sub_388FA10((unsigned __int64 *)&v80);
  }
  v114 = v51;
  v15 = sub_1646BA0(v51, 0);
  if ( sub_389BAC0((__int64 **)v6, v15, &v112, &v72, a3, 1) )
  {
LABEL_2:
    v8 = 1;
    goto LABEL_3;
  }
  v80 = v82;
  v83 = &v85;
  v81 = 0x800000000LL;
  v84 = 0x800000000LL;
  v17 = v51[2];
  v18 = (__int64 *)(v17 + 8);
  v48 = (__int64 *)(v17 + 8LL * *((unsigned int *)v51 + 3));
  if ( (_DWORD)v128 )
  {
    v62 = v6;
    v19 = (__int64 *)(v17 + 8);
    v45 = 24LL * (unsigned int)v128;
    v20 = 0;
    while ( 1 )
    {
      v43 = (int)v127;
      if ( v48 == v19 )
      {
        if ( !(*((_DWORD *)v51 + 2) >> 8) )
        {
          v78.m128i_i64[0] = (__int64)"too many arguments specified";
          v79 = 259;
          v8 = sub_38814C0(v62 + 8, *(_QWORD *)&v127[v20], (__int64)&v78);
          goto LABEL_69;
        }
        v22 = &v127[v20];
      }
      else
      {
        v21 = *v19;
        v22 = &v127[v20];
        if ( *v19 && v21 != **((_QWORD **)v22 + 1) )
        {
          sub_3888960((__int64 *)v76, v21);
          sub_95D570(v77, "argument is not of expected type '", (__int64)v76);
          sub_94F930(&v78, (__int64)v77, "'");
          v75 = 260;
          v74 = &v78;
          v8 = sub_38814C0(v62 + 8, *(_QWORD *)&v127[v20], (__int64)&v74);
          sub_2240A30((unsigned __int64 *)&v78);
          sub_2240A30((unsigned __int64 *)v77);
          sub_2240A30(v76);
          goto LABEL_69;
        }
        ++v19;
      }
      v23 = (unsigned int)v81;
      if ( (unsigned int)v81 >= HIDWORD(v81) )
      {
        sub_16CD150((__int64)&v80, v82, 0, 8, (int)v127, v16);
        v23 = (unsigned int)v81;
      }
      v80[v23] = *((_QWORD *)v22 + 1);
      LODWORD(v81) = v81 + 1;
      v24 = (unsigned int)v84;
      v25 = &v127[v20];
      if ( (unsigned int)v84 >= HIDWORD(v84) )
      {
        sub_16CD150((__int64)&v83, &v85, 0, 8, v43, v16);
        v24 = (unsigned int)v84;
      }
      v20 += 24;
      *(_QWORD *)&v83[8 * v24] = *((_QWORD *)v25 + 2);
      LODWORD(v84) = v84 + 1;
      if ( v45 == v20 )
      {
        v18 = v19;
        v6 = v62;
        break;
      }
    }
  }
  if ( v48 != v18 )
  {
    HIBYTE(v79) = 1;
    v31 = "not enough parameters specified for call";
LABEL_68:
    v78.m128i_i64[0] = (__int64)v31;
    LOBYTE(v79) = 3;
    v8 = sub_38814C0(v6 + 8, v55, (__int64)&v78);
    goto LABEL_69;
  }
  LOBYTE(v32) = sub_1560E20((__int64)&v98);
  v8 = v32;
  if ( (_BYTE)v32 )
  {
    HIBYTE(v79) = 1;
    v31 = "invoke instructions may not have an alignment";
    goto LABEL_68;
  }
  v56 = v83;
  v59 = (unsigned int)v84;
  v63 = sub_1560BF0(*(__int64 **)v6, &v87);
  v33 = sub_1560BF0(*(__int64 **)v6, &v98);
  v34 = 0;
  v49 = sub_155FDB0(*(__int64 **)v6, v33, v63, v56, v59);
  v79 = 257;
  v35 = v80;
  v64 = (unsigned int)v81;
  v66 = v71;
  v57 = v70;
  v50 = v72;
  for ( i = v109; &v109[7 * (unsigned int)v110] != i; i += 7 )
  {
    v37 = i[5] - i[4];
    v34 += v37 >> 3;
  }
  v44 = v109;
  v46 = (unsigned int)v110;
  v60 = v81 + v34 + 3;
  v38 = sub_1648AB0(72, v60, 16 * (int)v110);
  v39 = v60;
  v40 = (__int64)v38;
  if ( v38 )
  {
    v61 = v35;
    v41 = (__int64)v51;
    v53 = (__int64)v38;
    sub_15F1EA0((__int64)v38, **(_QWORD **)(v41 + 16), 5, (__int64)&v38[-3 * v39], v39, 0);
    *(_QWORD *)(v53 + 56) = 0;
    sub_15F6500(v53, v41, v50, v57, v66, (__int64)&v78, v61, v64, v44, v46);
    v40 = v53;
  }
  v65 = v40;
  *(_WORD *)(v40 + 18) = (4 * v67) | *(_WORD *)(v40 + 18) & 0x8003;
  *(_QWORD *)(v40 + 56) = v49;
  v78.m128i_i64[0] = v40;
  v42 = sub_3898320((_QWORD *)(v6 + 1128), (unsigned __int64 *)&v78);
  sub_3887600((__int64)v42, v73);
  *a2 = v65;
LABEL_69:
  if ( v83 != &v85 )
    _libc_free((unsigned __int64)v83);
  if ( v80 != v82 )
    _libc_free((unsigned __int64)v80);
LABEL_3:
  v9 = (unsigned __int64 *)v109;
  v10 = (unsigned __int64 *)&v109[7 * (unsigned int)v110];
  if ( v109 != (__int64 *)v10 )
  {
    do
    {
      v11 = *(v10 - 3);
      v10 -= 7;
      if ( v11 )
        j_j___libc_free_0(v11);
      if ( (unsigned __int64 *)*v10 != v10 + 2 )
        j_j___libc_free_0(*v10);
    }
    while ( v9 != v10 );
    v10 = (unsigned __int64 *)v109;
  }
  if ( v10 != (unsigned __int64 *)v111 )
    _libc_free((unsigned __int64)v10);
  if ( v127 != v129 )
    _libc_free((unsigned __int64)v127);
  if ( v126 )
    j_j___libc_free_0_0(v126);
  if ( v124 == sub_16982C0() )
  {
    v26 = v125;
    if ( v125 )
    {
      v27 = 32LL * *(_QWORD *)(v125 - 8);
      v28 = v125 + v27;
      if ( v125 != v125 + v27 )
      {
        do
        {
          v28 -= 32;
          sub_127D120((_QWORD *)(v28 + 8));
        }
        while ( v26 != v28 );
      }
      j_j_j___libc_free_0_0(v26 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v124);
  }
  if ( v122 > 0x40 && v121 )
    j_j___libc_free_0_0(v121);
  if ( v118 != (_QWORD *)v120 )
    j_j___libc_free_0((unsigned __int64)v118);
  if ( v115 != (_QWORD *)v117 )
    j_j___libc_free_0((unsigned __int64)v115);
  if ( v73[0] )
    j_j___libc_free_0((unsigned __int64)v73[0]);
  sub_3887AD0(v100);
  sub_3887AD0(v89);
  return v8;
}
