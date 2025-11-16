// Function: sub_377B740
// Address: 0x377b740
//
unsigned __int8 *__fastcall sub_377B740(
        __int64 a1,
        _QWORD *a2,
        unsigned int a3,
        __m128i a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned int v7; // r14d
  unsigned __int64 v8; // r15
  __int64 *v10; // rax
  unsigned __int16 *v11; // rdx
  __int64 v12; // rdi
  unsigned __int16 v13; // ax
  __int64 v14; // rdx
  unsigned __int16 *v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rsi
  int v19; // eax
  unsigned __int64 v20; // r13
  _OWORD *i; // rax
  _OWORD *v22; // rdx
  unsigned int v23; // eax
  unsigned __int64 v24; // rax
  unsigned int v25; // ebx
  __int64 v26; // r9
  __int64 *v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r12
  __int64 v30; // r13
  __int64 v31; // rdx
  unsigned __int16 *v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rax
  bool v35; // al
  __int64 v36; // r8
  int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // r8
  __int128 v40; // rax
  __int64 v41; // r9
  int v42; // edx
  unsigned __int8 *v43; // rax
  unsigned __int64 v44; // rcx
  unsigned __int8 *v45; // rbx
  unsigned __int8 *v46; // rdx
  unsigned __int8 *v47; // r12
  __int64 v48; // rax
  __int64 v49; // r8
  unsigned __int64 v50; // rdx
  unsigned __int8 **v51; // rax
  __int64 v52; // rax
  unsigned __int64 v53; // r12
  unsigned __int64 v54; // rdx
  unsigned __int8 **v55; // rax
  unsigned int v56; // r12d
  __int64 v57; // r8
  _QWORD *v58; // r14
  __int64 v59; // rdx
  __int64 v60; // r15
  __int64 v61; // rdx
  _QWORD *v62; // rdx
  unsigned __int8 *v63; // rax
  unsigned int v64; // edx
  __int64 *v65; // r12
  unsigned __int16 v66; // ax
  __int64 v67; // r9
  __int64 v68; // r8
  unsigned __int8 *v69; // r12
  __int64 v71; // rdx
  __int64 v72; // rdx
  __int128 v73; // [rsp-20h] [rbp-2E0h]
  __int128 v74; // [rsp-10h] [rbp-2D0h]
  __int128 v75; // [rsp-10h] [rbp-2D0h]
  __int128 v76; // [rsp-10h] [rbp-2D0h]
  __int64 v77; // [rsp-8h] [rbp-2C8h]
  __int64 v78; // [rsp+18h] [rbp-2A8h]
  __int64 v79; // [rsp+38h] [rbp-288h]
  unsigned int v80; // [rsp+40h] [rbp-280h]
  unsigned __int16 v81; // [rsp+46h] [rbp-27Ah]
  unsigned __int64 v82; // [rsp+48h] [rbp-278h]
  unsigned int v84; // [rsp+70h] [rbp-250h]
  unsigned int v85; // [rsp+74h] [rbp-24Ch]
  __int64 v86; // [rsp+80h] [rbp-240h]
  unsigned int v87; // [rsp+88h] [rbp-238h]
  __int64 v88; // [rsp+88h] [rbp-238h]
  __int64 v89; // [rsp+90h] [rbp-230h]
  _QWORD *v90; // [rsp+90h] [rbp-230h]
  int v91; // [rsp+98h] [rbp-228h]
  unsigned __int64 v92; // [rsp+98h] [rbp-228h]
  unsigned __int8 *v93; // [rsp+B0h] [rbp-210h]
  unsigned __int16 v94; // [rsp+D0h] [rbp-1F0h] BYREF
  __int64 v95; // [rsp+D8h] [rbp-1E8h]
  __int64 v96; // [rsp+E0h] [rbp-1E0h] BYREF
  int v97; // [rsp+E8h] [rbp-1D8h]
  __int64 v98; // [rsp+F0h] [rbp-1D0h] BYREF
  __int64 v99; // [rsp+F8h] [rbp-1C8h]
  unsigned __int16 v100; // [rsp+100h] [rbp-1C0h] BYREF
  __int64 v101; // [rsp+108h] [rbp-1B8h]
  __int16 v102; // [rsp+110h] [rbp-1B0h]
  __int64 v103; // [rsp+118h] [rbp-1A8h]
  _OWORD *v104; // [rsp+120h] [rbp-1A0h] BYREF
  __int64 v105; // [rsp+128h] [rbp-198h]
  _OWORD v106[4]; // [rsp+130h] [rbp-190h] BYREF
  _BYTE *v107; // [rsp+170h] [rbp-150h] BYREF
  __int64 v108; // [rsp+178h] [rbp-148h]
  _BYTE v109[128]; // [rsp+180h] [rbp-140h] BYREF
  _BYTE *v110; // [rsp+200h] [rbp-C0h] BYREF
  __int64 v111; // [rsp+208h] [rbp-B8h]
  _BYTE v112[176]; // [rsp+210h] [rbp-B0h] BYREF

  v8 = (unsigned __int64)a2;
  v10 = (__int64 *)a2[5];
  v11 = (unsigned __int16 *)a2[6];
  v12 = *v10;
  v79 = v10[1];
  v13 = *v11;
  v14 = *((_QWORD *)v11 + 1);
  v94 = v13;
  v95 = v14;
  if ( !v13 )
  {
    if ( !sub_3007100((__int64)&v94) )
      goto LABEL_55;
    goto LABEL_57;
  }
  if ( (unsigned __int16)(v13 - 176) <= 0x34u )
  {
LABEL_57:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( v94 )
    {
      if ( (unsigned __int16)(v94 - 176) > 0x34u )
      {
        v17 = v94 - 1;
        v84 = word_4456340[v17];
        goto LABEL_4;
      }
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
      goto LABEL_3;
    }
LABEL_55:
    v84 = sub_3007130((__int64)&v94, (__int64)a2);
    goto LABEL_56;
  }
LABEL_3:
  v15 = word_4456340;
  v16 = v94;
  v17 = v94 - 1;
  v84 = word_4456340[v17];
  if ( v94 )
  {
LABEL_4:
    v78 = 0;
    v81 = word_4456580[v17];
    goto LABEL_5;
  }
LABEL_56:
  v81 = sub_3009970((__int64)&v94, (__int64)a2, v16, (__int64)v15, a6);
  v78 = v71;
LABEL_5:
  v18 = a2[10];
  v96 = v18;
  if ( v18 )
    sub_B96E90((__int64)&v96, v18, 1);
  v19 = *(_DWORD *)(v8 + 72);
  v20 = *(unsigned int *)(v8 + 64);
  v105 = 0x400000000LL;
  v97 = v19;
  v107 = v109;
  v108 = 0x800000000LL;
  i = v106;
  v104 = v106;
  if ( v20 )
  {
    if ( v20 > 4 )
    {
      sub_C8D5F0((__int64)&v104, v106, v20, 0x10u, a6, a7);
      v22 = &v104[v20];
      for ( i = &v104[(unsigned int)v105]; v22 != i; ++i )
      {
LABEL_10:
        if ( i )
        {
          *(_QWORD *)i = 0;
          *((_DWORD *)i + 2) = 0;
        }
      }
    }
    else
    {
      v22 = &v106[v20];
      if ( v22 != v106 )
        goto LABEL_10;
    }
    LODWORD(v105) = v20;
  }
  if ( !a3 )
  {
    a7 = 1;
    v103 = 0;
    v102 = 1;
    v100 = v81;
    v101 = v78;
    v110 = v112;
    v111 = 0x800000000LL;
    if ( !v84 )
      goto LABEL_40;
    v80 = v84;
LABEL_18:
    v85 = 0;
    while ( 1 )
    {
      v24 = (unsigned __int64)v104;
      v25 = 1;
      *(_QWORD *)v104 = v12;
      *(_DWORD *)(v24 + 8) = v79;
      v91 = *(_DWORD *)(v8 + 64);
      if ( v91 != 1 )
        break;
LABEL_27:
      *((_QWORD *)&v74 + 1) = (unsigned int)v105;
      *(_QWORD *)&v74 = v104;
      v43 = sub_3411BE0(*(_QWORD **)(a1 + 8), *(_DWORD *)(v8 + 24), (__int64)&v96, &v100, 2, a7, v74);
      v44 = HIDWORD(v108);
      v45 = v43;
      v47 = v46;
      *((_DWORD *)v43 + 7) = *(_DWORD *)(v8 + 28);
      v48 = (unsigned int)v108;
      v49 = v77;
      v50 = (unsigned int)v108 + 1LL;
      if ( v50 > v44 )
      {
        sub_C8D5F0((__int64)&v107, v109, v50, 0x10u, v77, a7);
        v48 = (unsigned int)v108;
      }
      v51 = (unsigned __int8 **)&v107[16 * v48];
      v51[1] = v47;
      *v51 = v45;
      v52 = (unsigned int)v111;
      v53 = v82 & 0xFFFFFFFF00000000LL | 1;
      LODWORD(v108) = v108 + 1;
      v54 = (unsigned int)v111 + 1LL;
      v82 = v53;
      if ( v54 > HIDWORD(v111) )
      {
        sub_C8D5F0((__int64)&v110, v112, v54, 0x10u, v49, a7);
        v52 = (unsigned int)v111;
      }
      v55 = (unsigned __int8 **)&v110[16 * v52];
      ++v85;
      *v55 = v45;
      v55[1] = (unsigned __int8 *)v53;
      LODWORD(v111) = v111 + 1;
      if ( v80 == v85 )
      {
        if ( v84 <= v85 )
          goto LABEL_40;
        goto LABEL_33;
      }
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v26 = v25;
        v27 = (__int64 *)(*(_QWORD *)(v8 + 40) + 40LL * v25);
        v28 = *v27;
        v29 = *v27;
        v30 = v27[1];
        v31 = *((unsigned int *)v27 + 2);
        v32 = (unsigned __int16 *)(*(_QWORD *)(*v27 + 48) + 16 * v31);
        v33 = *v32;
        v34 = *((_QWORD *)v32 + 1);
        LOWORD(v98) = v33;
        v99 = v34;
        if ( !(_WORD)v33 )
          break;
        if ( (unsigned __int16)(v33 - 17) <= 0xD3u )
        {
          v39 = 0;
          LOWORD(v37) = word_4456580[(int)v33 - 1];
          goto LABEL_26;
        }
LABEL_22:
        a7 = (__int64)&v104[v26];
        ++v25;
        *(_QWORD *)a7 = v28;
        *(_DWORD *)(a7 + 8) = v31;
        if ( v25 == v91 )
          goto LABEL_27;
      }
      v87 = v31;
      v89 = v28;
      v35 = sub_30070B0((__int64)&v98);
      v28 = v89;
      LODWORD(v31) = v87;
      v26 = v25;
      if ( !v35 )
        goto LABEL_22;
      v37 = sub_3009970((__int64)&v98, v33, v87, v89, v36);
      v26 = v25;
      HIWORD(v7) = HIWORD(v37);
      v39 = v38;
LABEL_26:
      LOWORD(v7) = v37;
      ++v25;
      v86 = v26;
      v88 = v39;
      v90 = *(_QWORD **)(a1 + 8);
      *(_QWORD *)&v40 = sub_3400EE0((__int64)v90, v85, (__int64)&v96, 0, a4);
      *((_QWORD *)&v73 + 1) = v30;
      *(_QWORD *)&v73 = v29;
      v93 = sub_3406EB0(v90, 0x9Eu, (__int64)&v96, v7, v88, v41, v73, v40);
      a7 = (__int64)&v104[v86];
      *(_QWORD *)a7 = v93;
      *(_DWORD *)(a7 + 8) = v42;
      if ( v25 == v91 )
        goto LABEL_27;
    }
  }
  v23 = v84;
  v103 = 0;
  v102 = 1;
  v100 = v81;
  if ( v84 > a3 )
    v23 = a3;
  v84 = a3;
  v101 = v78;
  v110 = v112;
  v80 = v23;
  v111 = 0x800000000LL;
  if ( v23 )
    goto LABEL_18;
  v85 = 0;
LABEL_33:
  v92 = v8;
  v56 = v85;
  do
  {
    v98 = 0;
    LODWORD(v99) = 0;
    v58 = sub_33F17F0(*(_QWORD **)(a1 + 8), 51, (__int64)&v98, v81, v78);
    v60 = v59;
    if ( v98 )
      sub_B91220((__int64)&v98, v98);
    v61 = (unsigned int)v108;
    if ( (unsigned __int64)(unsigned int)v108 + 1 > HIDWORD(v108) )
    {
      sub_C8D5F0((__int64)&v107, v109, (unsigned int)v108 + 1LL, 0x10u, v57, a7);
      v61 = (unsigned int)v108;
    }
    v62 = &v107[16 * v61];
    ++v56;
    *v62 = v58;
    v62[1] = v60;
    LODWORD(v108) = v108 + 1;
  }
  while ( v56 < v84 );
  v8 = v92;
LABEL_40:
  *((_QWORD *)&v75 + 1) = (unsigned int)v111;
  *(_QWORD *)&v75 = v110;
  v63 = sub_33FC220(*(_QWORD **)(a1 + 8), 2, (__int64)&v96, 1, 0, a7, v75);
  sub_3760E70(a1, v8, 1, (unsigned __int64)v63, v64 | v79 & 0xFFFFFFFF00000000LL);
  v65 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
  v66 = sub_2D43050(v81, v84);
  v68 = 0;
  if ( !v66 )
  {
    v66 = sub_3009400(v65, v81, v78, v84, 0);
    v68 = v72;
  }
  *((_QWORD *)&v76 + 1) = (unsigned int)v108;
  *(_QWORD *)&v76 = v107;
  v69 = sub_33FC220(*(_QWORD **)(a1 + 8), 156, (__int64)&v96, v66, v68, v67, v76);
  if ( v110 != v112 )
    _libc_free((unsigned __int64)v110);
  if ( v104 != v106 )
    _libc_free((unsigned __int64)v104);
  if ( v107 != v109 )
    _libc_free((unsigned __int64)v107);
  if ( v96 )
    sub_B91220((__int64)&v96, v96);
  return v69;
}
