// Function: sub_12351A0
// Address: 0x12351a0
//
__int64 __fastcall sub_12351A0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 **v4; // rdx
  __int64 *v5; // rdx
  _DWORD *v6; // r13
  __int64 v7; // rsi
  _QWORD *v8; // rbx
  _QWORD *v9; // r12
  __int64 v10; // rdi
  void *v11; // r14
  unsigned __int64 v13; // rax
  __int64 v14; // rsi
  void **v15; // r13
  void **v16; // rbx
  __int64 *v17; // rdi
  int v18; // r9d
  __int64 v19; // rax
  __int64 *v20; // rdx
  unsigned __int64 *v21; // rbx
  __int64 v22; // rsi
  __int64 v23; // r8
  __int64 *v24; // r12
  __int64 v25; // rdx
  unsigned __int64 v26; // r9
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 **v29; // rbx
  __int64 *v30; // rdi
  unsigned __int64 v31; // rax
  int v32; // edi
  _BYTE *i; // rdx
  __int64 v34; // rsi
  _QWORD *v35; // rax
  unsigned __int64 *v36; // rax
  __int64 v37; // [rsp+0h] [rbp-5E0h]
  __int64 v38; // [rsp+8h] [rbp-5D8h]
  unsigned int v39; // [rsp+14h] [rbp-5CCh]
  __int64 v40; // [rsp+20h] [rbp-5C0h]
  unsigned __int64 v41; // [rsp+28h] [rbp-5B8h]
  unsigned __int64 *v42; // [rsp+30h] [rbp-5B0h]
  __int64 v43; // [rsp+30h] [rbp-5B0h]
  unsigned __int64 v44; // [rsp+40h] [rbp-5A0h]
  __int64 v45; // [rsp+48h] [rbp-598h]
  unsigned __int64 v47; // [rsp+58h] [rbp-588h]
  __int64 v48; // [rsp+58h] [rbp-588h]
  __int64 v49; // [rsp+60h] [rbp-580h]
  __int64 **v50; // [rsp+68h] [rbp-578h]
  char *v51; // [rsp+68h] [rbp-578h]
  __int64 v52; // [rsp+68h] [rbp-578h]
  __int64 *v54; // [rsp+70h] [rbp-570h]
  unsigned __int64 v55; // [rsp+70h] [rbp-570h]
  __int64 *v56; // [rsp+70h] [rbp-570h]
  unsigned __int64 v57; // [rsp+80h] [rbp-560h]
  __int64 v58; // [rsp+80h] [rbp-560h]
  unsigned __int8 v59; // [rsp+A8h] [rbp-538h]
  int v60; // [rsp+B8h] [rbp-528h] BYREF
  int v61; // [rsp+BCh] [rbp-524h] BYREF
  unsigned __int64 v62; // [rsp+C0h] [rbp-520h] BYREF
  __int64 *v63; // [rsp+C8h] [rbp-518h] BYREF
  __int64 v64; // [rsp+D0h] [rbp-510h] BYREF
  __int64 v65; // [rsp+D8h] [rbp-508h] BYREF
  unsigned __int64 v66; // [rsp+E0h] [rbp-500h] BYREF
  __int64 v67; // [rsp+E8h] [rbp-4F8h] BYREF
  char *v68[2]; // [rsp+F0h] [rbp-4F0h] BYREF
  __int64 v69; // [rsp+100h] [rbp-4E0h]
  __int64 v70[4]; // [rsp+110h] [rbp-4D0h] BYREF
  __m128i v71[2]; // [rsp+130h] [rbp-4B0h] BYREF
  __m128i v72[2]; // [rsp+150h] [rbp-490h] BYREF
  unsigned __int64 v73[4]; // [rsp+170h] [rbp-470h] BYREF
  __int16 v74; // [rsp+190h] [rbp-450h]
  __int64 *v75; // [rsp+1A0h] [rbp-440h] BYREF
  __int64 v76; // [rsp+1A8h] [rbp-438h]
  _BYTE v77[64]; // [rsp+1B0h] [rbp-430h] BYREF
  const char *v78; // [rsp+1F0h] [rbp-3F0h] BYREF
  __int64 v79; // [rsp+1F8h] [rbp-3E8h]
  _BYTE v80[16]; // [rsp+200h] [rbp-3E0h] BYREF
  char v81; // [rsp+210h] [rbp-3D0h]
  char v82; // [rsp+211h] [rbp-3CFh]
  __int64 *v83; // [rsp+240h] [rbp-3A0h] BYREF
  _BYTE *v84; // [rsp+248h] [rbp-398h]
  __int64 v85; // [rsp+250h] [rbp-390h]
  _BYTE v86[72]; // [rsp+258h] [rbp-388h] BYREF
  __int64 *v87; // [rsp+2A0h] [rbp-340h] BYREF
  _BYTE *v88; // [rsp+2A8h] [rbp-338h]
  __int64 v89; // [rsp+2B0h] [rbp-330h]
  _BYTE v90[72]; // [rsp+2B8h] [rbp-328h] BYREF
  _BYTE *v91; // [rsp+300h] [rbp-2E0h] BYREF
  __int64 v92; // [rsp+308h] [rbp-2D8h]
  _BYTE v93[112]; // [rsp+310h] [rbp-2D0h] BYREF
  unsigned int v94; // [rsp+380h] [rbp-260h] BYREF
  __int64 v95; // [rsp+388h] [rbp-258h]
  unsigned __int64 v96; // [rsp+398h] [rbp-248h]
  _QWORD *v97; // [rsp+3A0h] [rbp-240h]
  __int64 v98; // [rsp+3A8h] [rbp-238h]
  _QWORD v99[2]; // [rsp+3B0h] [rbp-230h] BYREF
  _QWORD *v100; // [rsp+3C0h] [rbp-220h]
  __int64 v101; // [rsp+3C8h] [rbp-218h]
  _QWORD v102[2]; // [rsp+3D0h] [rbp-210h] BYREF
  __int64 v103; // [rsp+3E0h] [rbp-200h]
  unsigned int v104; // [rsp+3E8h] [rbp-1F8h]
  char v105; // [rsp+3ECh] [rbp-1F4h]
  void *v106; // [rsp+3F0h] [rbp-1F0h] BYREF
  void **v107; // [rsp+3F8h] [rbp-1E8h]
  __int64 v108; // [rsp+410h] [rbp-1D0h]
  char v109; // [rsp+418h] [rbp-1C8h]
  unsigned __int64 *v110; // [rsp+420h] [rbp-1C0h] BYREF
  __int64 v111; // [rsp+428h] [rbp-1B8h]
  _BYTE v112[432]; // [rsp+430h] [rbp-1B0h] BYREF

  v4 = *(__int64 ***)(a1 + 344);
  v47 = *(_QWORD *)(a1 + 232);
  v83 = *v4;
  v84 = v86;
  v85 = 0x800000000LL;
  v5 = *v4;
  v89 = 0x800000000LL;
  v88 = v90;
  v87 = v5;
  v68[0] = 0;
  v68[1] = 0;
  v69 = 0;
  v62 = 0;
  v63 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = v99;
  v98 = 0;
  LOBYTE(v99[0]) = 0;
  v100 = v102;
  v101 = 0;
  LOBYTE(v102[0]) = 0;
  v104 = 1;
  v103 = 0;
  v105 = 0;
  v6 = sub_C33320();
  sub_C3B1B0((__int64)&v110, 0.0);
  sub_C407B0(&v106, (__int64 *)&v110, v6);
  sub_C338F0((__int64)&v110);
  v7 = (__int64)&v60;
  v111 = 0x1000000000LL;
  v108 = 0;
  v109 = 0;
  v110 = (unsigned __int64 *)v112;
  v91 = v93;
  v92 = 0x200000000LL;
  if ( (unsigned __int8)sub_120C5E0(a1, &v60) )
    goto LABEL_2;
  v7 = (__int64)&v83;
  if ( (unsigned __int8)sub_1218580(a1, &v83, 0) )
    goto LABEL_2;
  v7 = (__int64)&v61;
  if ( (unsigned __int8)sub_1212650(a1, &v61, *(_DWORD *)(*(_QWORD *)(a1 + 344) + 320LL)) )
    goto LABEL_2;
  v13 = *(_QWORD *)(a1 + 232);
  v7 = (__int64)&v63;
  v82 = 1;
  v44 = v13;
  v78 = "expected type";
  v81 = 3;
  if ( (unsigned __int8)sub_12190A0(a1, &v63, (int *)&v78, 1) )
    goto LABEL_2;
  v7 = (__int64)&v94;
  if ( (unsigned __int8)sub_1221570((_QWORD **)a1, (__int64)&v94, (__int64)a3, 0) )
    goto LABEL_2;
  v7 = (__int64)&v110;
  if ( (unsigned __int8)sub_122F150(a1, (__int64)&v110, a3, 0, 0) )
    goto LABEL_2;
  v7 = (__int64)&v87;
  if ( (unsigned __int8)sub_1218010(a1, &v87, (__int64)v68, 0, &v62) )
    goto LABEL_2;
  v7 = (__int64)&v91;
  if ( (unsigned __int8)sub_122F1C0(a1, (__int64)&v91, a3) )
    goto LABEL_2;
  v7 = 56;
  if ( (unsigned __int8)sub_120AFE0(a1, 56, "expected 'to' in invoke") )
    goto LABEL_2;
  v7 = (__int64)&v64;
  v78 = 0;
  if ( (unsigned __int8)sub_122FEA0(a1, &v64, (unsigned __int64 *)&v78, a3) )
    goto LABEL_2;
  v7 = 66;
  if ( (unsigned __int8)sub_120AFE0(a1, 66, "expected 'unwind' in invoke") )
    goto LABEL_2;
  v7 = (__int64)&v65;
  v78 = 0;
  if ( (unsigned __int8)sub_122FEA0(a1, &v65, (unsigned __int64 *)&v78, a3) )
    goto LABEL_2;
  v59 = sub_12104A0(a1, v63, (__int64)v110, (unsigned int)v111, &v66);
  if ( v59 )
  {
    v7 = v44;
    v78 = "Invalid result type for LLVM function";
    v82 = 1;
    v81 = 3;
    sub_11FD800(a1 + 176, v44, (__int64)&v78, 1);
    goto LABEL_3;
  }
  v17 = *(__int64 **)a1;
  v96 = v66;
  v7 = sub_BCE3C0(v17, v61);
  v59 = sub_121E800((__int64 **)a1, v7, &v94, &v67, a3, v18);
  if ( v59 )
  {
LABEL_2:
    v59 = 1;
    goto LABEL_3;
  }
  v50 = (__int64 **)a1;
  v78 = v80;
  v75 = (__int64 *)v77;
  v76 = 0x800000000LL;
  v79 = 0x800000000LL;
  v19 = *(_QWORD *)(v66 + 16);
  v20 = (__int64 *)(v19 + 8);
  v54 = (__int64 *)(v19 + 8LL * *(unsigned int *)(v66 + 12));
  v21 = v110;
  v42 = &v110[3 * (unsigned int)v111];
  while ( v21 != v42 )
  {
    if ( v20 == v54 )
    {
      if ( !(*(_DWORD *)(v66 + 8) >> 8) )
      {
        v73[0] = (unsigned __int64)"too many arguments specified";
        v74 = 259;
        v7 = *v21;
        sub_11FD800((__int64)(v50 + 22), *v21, (__int64)v73, 1);
        v59 = 1;
        goto LABEL_64;
      }
      v24 = v54;
      v23 = v21[1];
    }
    else
    {
      v22 = *v20;
      v23 = v21[1];
      v24 = v20 + 1;
      if ( *v20 && v22 != *(_QWORD *)(v23 + 8) )
      {
        sub_1207630(v70, v22);
        sub_95D570(v71, "argument is not of expected type '", (__int64)v70);
        sub_94F930(v72, (__int64)v71, "'");
        v74 = 260;
        v73[0] = (unsigned __int64)v72;
        v7 = *v21;
        sub_11FD800((__int64)(v50 + 22), *v21, (__int64)v73, 1);
        sub_2240A30(v72);
        sub_2240A30(v71);
        sub_2240A30(v70);
        v59 = 1;
        goto LABEL_64;
      }
    }
    v25 = (unsigned int)v76;
    v26 = (unsigned int)v76 + 1LL;
    if ( v26 > HIDWORD(v76) )
    {
      v40 = v23;
      sub_C8D5F0((__int64)&v75, v77, (unsigned int)v76 + 1LL, 8u, v23, v26);
      v25 = (unsigned int)v76;
      v23 = v40;
    }
    v27 = (__int64)v75;
    v21 += 3;
    v75[v25] = v23;
    v28 = *(v21 - 1);
    LODWORD(v76) = v76 + 1;
    sub_1212C70((__int64)&v78, v28, v25, v27, v23, v26);
    v20 = v24;
  }
  v29 = v50;
  if ( v20 == v54 )
  {
    v30 = *v50;
    v51 = (char *)v78;
    v55 = (unsigned int)v79;
    v57 = sub_A7A280(v30, (__int64)&v83);
    v31 = sub_A7A280(*v29, (__int64)&v87);
    v32 = 0;
    v41 = sub_A78180(*v29, v31, v57, v51, v55);
    v56 = v75;
    v74 = 257;
    v43 = (unsigned int)v76;
    v49 = v65;
    v48 = v64;
    v45 = v67;
    v52 = v66;
    for ( i = v91; &v91[56 * (unsigned int)v92] != i; i += 56 )
    {
      v34 = *((_QWORD *)i + 5) - *((_QWORD *)i + 4);
      v32 += v34 >> 3;
    }
    v37 = (__int64)v91;
    LOBYTE(v40) = 16 * (_DWORD)v92 != 0;
    v39 = v76 + v32 + 3;
    v38 = (unsigned int)v92;
    v35 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v92) << 32) | v39);
    v58 = (__int64)v35;
    if ( v35 )
    {
      sub_B44260((__int64)v35, **(_QWORD **)(v52 + 16), 5, v39 & 0x7FFFFFF | ((_DWORD)v40 << 28), 0, 0);
      *(_QWORD *)(v58 + 72) = 0;
      sub_B4A9C0(v58, v52, v45, v48, v49, (__int64)v73, v56, v43, v37, v38);
    }
    *(_WORD *)(v58 + 2) = (4 * v60) | *(_WORD *)(v58 + 2) & 0xF003;
    *(_QWORD *)(v58 + 72) = v41;
    v73[0] = v58;
    v36 = sub_121BCE0(v29 + 179, v73);
    v7 = (__int64)v68;
    sub_1205F70((__int64)v36, v68);
    *a2 = v58;
  }
  else
  {
    v7 = v47;
    v73[0] = (unsigned __int64)"not enough parameters specified for call";
    v74 = 259;
    sub_11FD800((__int64)(v50 + 22), v47, (__int64)v73, 1);
    v59 = 1;
  }
LABEL_64:
  if ( v78 != v80 )
    _libc_free(v78, v7);
  if ( v75 != (__int64 *)v77 )
    _libc_free(v75, v7);
LABEL_3:
  v8 = v91;
  v9 = &v91[56 * (unsigned int)v92];
  if ( v91 != (_BYTE *)v9 )
  {
    do
    {
      v10 = *(v9 - 3);
      v9 -= 7;
      if ( v10 )
      {
        v7 = v9[6] - v10;
        j_j___libc_free_0(v10, v7);
      }
      if ( (_QWORD *)*v9 != v9 + 2 )
      {
        v7 = v9[2] + 1LL;
        j_j___libc_free_0(*v9, v7);
      }
    }
    while ( v8 != v9 );
    v9 = v91;
  }
  if ( v9 != (_QWORD *)v93 )
    _libc_free(v9, v7);
  if ( v110 != (unsigned __int64 *)v112 )
    _libc_free(v110, v7);
  if ( v108 )
    j_j___libc_free_0_0(v108);
  v11 = sub_C33340();
  if ( v106 == v11 )
  {
    if ( v107 )
    {
      v14 = 3LL * (_QWORD)*(v107 - 1);
      v15 = &v107[v14];
      if ( v107 == &v107[v14] )
      {
        v7 = v14 * 8 + 8;
        j_j_j___libc_free_0_0(v15 - 1);
      }
      else
      {
        do
        {
          while ( 1 )
          {
            v16 = v15;
            v15 -= 3;
            if ( v11 == *v15 )
              break;
            sub_C338F0((__int64)v15);
            if ( v107 == v15 )
              goto LABEL_51;
          }
          sub_969EE0((__int64)v15);
        }
        while ( v107 != v15 );
LABEL_51:
        v7 = 24LL * (_QWORD)*(v16 - 4) + 8;
        j_j_j___libc_free_0_0(v15 - 1);
      }
    }
  }
  else
  {
    sub_C338F0((__int64)&v106);
  }
  if ( v104 > 0x40 && v103 )
    j_j___libc_free_0_0(v103);
  if ( v100 != v102 )
  {
    v7 = v102[0] + 1LL;
    j_j___libc_free_0(v100, v102[0] + 1LL);
  }
  if ( v97 != v99 )
  {
    v7 = v99[0] + 1LL;
    j_j___libc_free_0(v97, v99[0] + 1LL);
  }
  if ( v68[0] )
  {
    v7 = v69 - (unsigned __int64)v68[0];
    j_j___libc_free_0(v68[0], v69 - (unsigned __int64)v68[0]);
  }
  if ( v88 != v90 )
    _libc_free(v88, v7);
  if ( v84 != v86 )
    _libc_free(v84, v7);
  return v59;
}
