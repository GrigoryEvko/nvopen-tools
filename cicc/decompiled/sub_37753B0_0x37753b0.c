// Function: sub_37753B0
// Address: 0x37753b0
//
unsigned __int8 *__fastcall sub_37753B0(
        _QWORD *a1,
        __int64 a2,
        unsigned int **a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        __int64 a9,
        unsigned int a10,
        __int64 a11)
{
  unsigned int *v11; // rdx
  unsigned __int16 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rsi
  unsigned __int16 v16; // bx
  unsigned int *v17; // rcx
  __int64 v18; // rax
  unsigned __int16 *v19; // rdx
  __int64 v20; // rdx
  unsigned __int8 *v21; // r12
  unsigned int v22; // r12d
  unsigned int v23; // ecx
  unsigned int v24; // ebx
  _QWORD *v25; // r12
  unsigned int v26; // edx
  unsigned int v27; // r13d
  __int64 v28; // rdx
  unsigned int *v29; // rcx
  __int64 v31; // rax
  __int16 v32; // dx
  __int64 v33; // rsi
  int v34; // r12d
  unsigned int *v35; // rcx
  __int64 v36; // rax
  int v37; // ebx
  __int64 v38; // r14
  unsigned int v39; // r15d
  __int64 *v40; // r13
  __int64 v41; // rax
  __int64 v42; // rdx
  __int16 v43; // ecx^2
  __int64 v44; // rbx
  _QWORD *v45; // rax
  unsigned __int64 v46; // rdx
  __int64 v47; // r12
  unsigned __int64 v48; // r13
  __int64 v49; // r9
  __int64 v50; // rbx
  __int128 v51; // rax
  __int64 v52; // r9
  unsigned int v53; // edx
  __int64 v54; // rax
  _QWORD *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rsi
  _QWORD *v58; // r13
  __int64 v59; // r8
  bool v60; // al
  unsigned int v61; // eax
  unsigned int v62; // ecx
  __int64 v63; // rax
  unsigned int v64; // r15d
  _BYTE *v65; // rcx
  _BYTE *v66; // rdx
  _BYTE *v67; // rax
  __int64 v68; // rax
  __int64 v69; // r12
  __int64 v70; // rcx
  _BYTE *v71; // rdx
  unsigned int *v72; // rcx
  unsigned __int8 *v73; // rax
  int v74; // edx
  int v75; // edi
  unsigned __int8 *v76; // rdx
  __int64 v77; // rax
  unsigned int v78; // ecx
  _BYTE *v79; // rbx
  __int64 v80; // rdx
  __int128 v81; // [rsp-30h] [rbp-260h]
  __int128 v82; // [rsp-10h] [rbp-240h]
  __int128 v83; // [rsp-10h] [rbp-240h]
  __int64 v84; // [rsp-8h] [rbp-238h]
  int v85; // [rsp+8h] [rbp-228h]
  __int64 v86; // [rsp+10h] [rbp-220h]
  __int64 v87; // [rsp+18h] [rbp-218h]
  int v89; // [rsp+38h] [rbp-1F8h]
  __int64 v90; // [rsp+38h] [rbp-1F8h]
  __int64 v91; // [rsp+40h] [rbp-1F0h]
  __int64 v92; // [rsp+40h] [rbp-1F0h]
  __int64 v93; // [rsp+40h] [rbp-1F0h]
  __int64 v94; // [rsp+40h] [rbp-1F0h]
  unsigned int v95; // [rsp+48h] [rbp-1E8h]
  int v96; // [rsp+4Ch] [rbp-1E4h]
  int v97; // [rsp+50h] [rbp-1E0h]
  unsigned __int16 v98; // [rsp+54h] [rbp-1DCh]
  __int16 v99; // [rsp+56h] [rbp-1DAh]
  __int64 v100; // [rsp+58h] [rbp-1D8h]
  unsigned int v101; // [rsp+60h] [rbp-1D0h]
  _QWORD *v102; // [rsp+60h] [rbp-1D0h]
  __int64 v103; // [rsp+60h] [rbp-1D0h]
  unsigned __int64 v104; // [rsp+68h] [rbp-1C8h]
  __int64 v107; // [rsp+D0h] [rbp-160h] BYREF
  __int64 v108; // [rsp+D8h] [rbp-158h]
  __int64 v109; // [rsp+E0h] [rbp-150h] BYREF
  int v110; // [rsp+E8h] [rbp-148h]
  _BYTE *v111; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v112; // [rsp+F8h] [rbp-138h]
  _BYTE v113[304]; // [rsp+100h] [rbp-130h] BYREF

  v11 = *a3;
  v95 = a4;
  v107 = a5;
  v108 = a6;
  if ( (_DWORD)a4 == 1 )
  {
    v12 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v11 + 48LL) + 16LL * v11[2]);
    a4 = *v12;
    v13 = *((_QWORD *)v12 + 1);
    LOWORD(v107) = a4;
    v108 = v13;
    if ( (_WORD)a4 == (_WORD)a10 && (a11 == v13 || (_WORD)a4) )
      return *(unsigned __int8 **)v11;
  }
  v14 = *(_QWORD *)v11;
  v15 = *(_QWORD *)(*(_QWORD *)v11 + 80LL);
  v109 = v15;
  if ( v15 )
    sub_B96E90((__int64)&v109, v15, 1);
  v110 = *(_DWORD *)(v14 + 72);
  if ( (_WORD)a10 )
  {
    v86 = 0;
    v98 = word_4456580[(unsigned __int16)a10 - 1];
  }
  else
  {
    v98 = sub_3009970((__int64)&a10, v15, (__int64)v11, a4, a5);
    v86 = v80;
  }
  v96 = v95 - 1;
  while ( 1 )
  {
    v16 = a8;
    v17 = *a3;
    v18 = *(_QWORD *)(*(_QWORD *)&(*a3)[4 * v96] + 48LL) + 16LL * (*a3)[4 * v96 + 2];
    if ( (_WORD)a8 == *(_WORD *)v18 && (*(_QWORD *)(v18 + 8) == a9 || (_WORD)a8) )
      break;
    v89 = v96;
    v31 = *(_QWORD *)(*(_QWORD *)&v17[4 * v96] + 48LL) + 16LL * v17[4 * v96 + 2];
    v32 = *(_WORD *)v31;
    v33 = *(_QWORD *)(v31 + 8);
    LOWORD(v107) = *(_WORD *)v31;
    v108 = v33;
    v34 = v95 - 2;
    if ( (int)(v95 - 2) < 0 )
    {
      v37 = 1;
      v97 = v96;
      goto LABEL_36;
    }
    v35 = &v17[4 * v34];
    while ( 1 )
    {
      v36 = *(_QWORD *)(*(_QWORD *)v35 + 48LL) + 16LL * v35[2];
      if ( v32 != *(_WORD *)v36 )
      {
        v95 = v34 + 2;
        v89 = v34 + 1;
        v37 = v96 - v34;
        v97 = v34 + 1;
        v96 = v34 + 1;
        goto LABEL_36;
      }
      if ( v33 != *(_QWORD *)(v36 + 8) && !v32 )
        break;
      v35 -= 4;
      if ( !v34 )
      {
        v97 = 0;
        v37 = v95;
        v89 = 0;
        v34 = -1;
        v95 = 1;
        v96 = 0;
LABEL_36:
        if ( !v32 )
          goto LABEL_78;
        if ( (unsigned __int16)(v32 - 17) <= 0xD3u )
        {
          if ( (unsigned __int16)(v32 - 176) <= 0x34u )
            goto LABEL_102;
          goto LABEL_99;
        }
LABEL_38:
        v101 = v37;
        v38 = v87;
        v39 = 1;
        goto LABEL_39;
      }
      --v34;
    }
    v95 = v34 + 2;
    v89 = v34 + 1;
    v37 = v96 - v34;
    v97 = v34 + 1;
    v96 = v34 + 1;
LABEL_78:
    if ( !sub_30070B0((__int64)&v107) )
      goto LABEL_38;
    if ( !sub_3007100((__int64)&v107) )
      goto LABEL_80;
LABEL_102:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !(_WORD)v107 )
    {
LABEL_80:
      v39 = sub_3007130((__int64)&v107, v33);
      goto LABEL_81;
    }
    if ( (unsigned __int16)(v107 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_99:
    v39 = word_4456340[(unsigned __int16)v107 - 1];
LABEL_81:
    v101 = v37;
    v38 = v87;
LABEL_39:
    while ( 2 )
    {
      while ( 1 )
      {
        v39 *= 2;
        v40 = (__int64 *)a1[8];
        LOWORD(v41) = sub_2D43050(v98, v39);
        if ( (_WORD)v41 )
          break;
        v41 = sub_3009400(v40, v98, v86, v39, 0);
        v38 = v41;
        v43 = WORD1(v41);
        if ( (_WORD)v41 )
          goto LABEL_41;
      }
      LOWORD(v38) = v41;
      v42 = 0;
      v43 = WORD1(v38);
LABEL_41:
      if ( !*(_QWORD *)(a2 + 8LL * (unsigned __int16)v41 + 112) )
        continue;
      break;
    }
    v99 = v41;
    v87 = v38;
    v44 = v101;
    WORD1(v38) = v43;
    v100 = v42;
    if ( (_WORD)v107 )
    {
      if ( (unsigned __int16)(v107 - 17) > 0xD3u )
        goto LABEL_44;
LABEL_50:
      v111 = 0;
      LODWORD(v112) = 0;
      v55 = sub_33F17F0(a1, 51, (__int64)&v111, v107, v108);
      v57 = (__int64)v111;
      v58 = v55;
      v59 = v56;
      if ( v111 )
      {
        v92 = v56;
        sub_B91220((__int64)&v111, (__int64)v111);
        v59 = v92;
      }
      if ( (_WORD)v107 )
      {
        if ( (unsigned __int16)(v107 - 176) <= 0x34u )
          goto LABEL_96;
      }
      else
      {
        v90 = v59;
        v60 = sub_3007100((__int64)&v107);
        v59 = v90;
        if ( !v60 )
          goto LABEL_54;
LABEL_96:
        v94 = v59;
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
        v59 = v94;
        if ( !(_WORD)v107 )
        {
LABEL_54:
          v93 = v59;
          v61 = sub_3007130((__int64)&v107, v57);
          v59 = v93;
          v62 = v61;
          goto LABEL_55;
        }
        if ( (unsigned __int16)(v107 - 176) <= 0x34u )
        {
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
          v59 = v94;
        }
      }
      v62 = word_4456340[(unsigned __int16)v107 - 1];
LABEL_55:
      v111 = v113;
      v112 = 0x1000000000LL;
      v64 = v39 / v62;
      v63 = v64;
      if ( v64 )
      {
        v65 = v113;
        v66 = v113;
        if ( v64 > 0x10uLL )
        {
          v85 = v59;
          sub_C8D5F0((__int64)&v111, v113, v64, 0x10u, v59, (__int64)v113);
          v66 = v111;
          LODWORD(v59) = v85;
          v63 = v64;
          v65 = &v111[16 * (unsigned int)v112];
        }
        v67 = &v66[16 * v63];
        if ( v67 != v65 )
        {
          do
          {
            if ( v65 )
            {
              *(_QWORD *)v65 = 0;
              *((_DWORD *)v65 + 2) = 0;
            }
            v65 += 16;
          }
          while ( v67 != v65 );
          v66 = v111;
        }
        LODWORD(v112) = v64;
        if ( v101 )
        {
LABEL_64:
          v68 = 0;
          v69 = 4 * (v34 + 1LL);
          while ( 1 )
          {
            v70 = v68++;
            v71 = &v66[16 * v70];
            v72 = *a3;
            *(_QWORD *)v71 = *(_QWORD *)&(*a3)[v69];
            LODWORD(v72) = v72[v69 + 2];
            v69 += 4;
            *((_DWORD *)v71 + 2) = (_DWORD)v72;
            if ( v101 <= (unsigned int)v68 )
              break;
            v66 = v111;
          }
          v66 = v111;
          if ( v101 >= v64 )
          {
LABEL_68:
            v63 = (unsigned int)v112;
            goto LABEL_69;
          }
        }
        while ( 1 )
        {
          v78 = v44 + 1;
          v79 = &v66[16 * v44];
          *(_QWORD *)v79 = v58;
          *((_DWORD *)v79 + 2) = v59;
          v66 = v111;
          if ( v64 <= v78 )
            break;
          v44 = v78;
        }
        goto LABEL_68;
      }
      v66 = v113;
      if ( v101 )
        goto LABEL_64;
LABEL_69:
      LOWORD(v38) = v99;
      *((_QWORD *)&v83 + 1) = v63;
      *(_QWORD *)&v83 = v66;
      v73 = sub_33FC220(a1, 159, (__int64)&v109, (unsigned int)v38, v100, (__int64)v113, v83);
      v75 = v74;
      v76 = v73;
      v77 = (__int64)&(*a3)[4 * v97];
      *(_QWORD *)v77 = v76;
      *(_DWORD *)(v77 + 8) = v75;
      v15 = v84;
      if ( v111 != v113 )
        _libc_free((unsigned __int64)v111);
    }
    else
    {
      if ( sub_30070B0((__int64)&v107) )
        goto LABEL_50;
LABEL_44:
      LOWORD(v38) = v99;
      v111 = 0;
      LODWORD(v112) = 0;
      v45 = sub_33F17F0(a1, 51, (__int64)&v111, v38, v100);
      v15 = (__int64)v111;
      if ( v111 )
      {
        v102 = v45;
        v104 = v46;
        sub_B91220((__int64)&v111, (__int64)v111);
        v45 = v102;
        v46 = v104;
      }
      v47 = (__int64)v45;
      v48 = v46;
      v49 = 0;
      v91 = (unsigned int)v44;
      if ( (_DWORD)v44 )
      {
        do
        {
          v103 = v49;
          v50 = (unsigned int)(v97 + v49);
          *(_QWORD *)&v51 = sub_3400EE0((__int64)a1, v49, (__int64)&v109, 0, a7);
          LOWORD(v38) = v99;
          v15 = 157;
          *((_QWORD *)&v81 + 1) = v48;
          *(_QWORD *)&v81 = v47;
          v47 = sub_340F900(a1, 0x9Du, (__int64)&v109, v38, v100, v52, v81, *(_OWORD *)&(*a3)[4 * v50], v51);
          v49 = v103 + 1;
          v48 = v53 | v48 & 0xFFFFFFFF00000000LL;
        }
        while ( v91 != v103 + 1 );
      }
      v54 = (__int64)&(*a3)[4 * v89];
      *(_QWORD *)v54 = v47;
      *(_DWORD *)(v54 + 8) = v48;
    }
  }
  if ( v95 != 1
    || (v19 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v17 + 48LL) + 16LL * v17[2]),
        v15 = *v19,
        v20 = *((_QWORD *)v19 + 1),
        LOWORD(v107) = v15,
        v108 = v20,
        (_WORD)v15 != (_WORD)a10) )
  {
    if ( (_WORD)a10 )
    {
      if ( (unsigned __int16)(a10 - 176) > 0x34u )
        goto LABEL_18;
      goto LABEL_100;
    }
LABEL_87:
    if ( !sub_3007100((__int64)&a10) )
      goto LABEL_88;
LABEL_100:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !(_WORD)a10 )
    {
LABEL_88:
      v22 = sub_3007130((__int64)&a10, v15);
      if ( !v16 )
        goto LABEL_89;
      goto LABEL_19;
    }
    if ( (unsigned __int16)(a10 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_18:
    v22 = word_4456340[(unsigned __int16)a10 - 1];
    if ( !v16 )
    {
LABEL_89:
      if ( sub_3007100((__int64)&a8) )
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
      v23 = sub_3007130((__int64)&a8, v15);
LABEL_22:
      v24 = v22 / v23;
      if ( v95 != v22 / v23 )
      {
        v111 = 0;
        LODWORD(v112) = 0;
        v25 = sub_33F17F0(a1, 51, (__int64)&v111, a8, a9);
        v27 = v26;
        if ( v111 )
          sub_B91220((__int64)&v111, (__int64)v111);
        if ( v95 < v24 )
        {
          v28 = 4LL * v95;
          do
          {
            v29 = *a3;
            *(_QWORD *)&v29[v28] = v25;
            v29[v28 + 2] = v27;
            v28 += 4;
          }
          while ( v28 != 4 * (v95 + (unsigned __int64)(v24 + ~v95) + 1) );
        }
      }
      *((_QWORD *)&v82 + 1) = v24;
      *(_QWORD *)&v82 = *a3;
      v21 = sub_33FC220(a1, 159, (__int64)&v109, a10, a11, (__int64)&v109, v82);
      goto LABEL_29;
    }
LABEL_19:
    if ( (unsigned __int16)(v16 - 176) <= 0x34u )
    {
      sub_CA17B0(
        "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT"
        "::getVectorElementCount() instead");
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    }
    v23 = word_4456340[v16 - 1];
    goto LABEL_22;
  }
  if ( a11 != v20 && !(_WORD)a10 )
    goto LABEL_87;
  v21 = *(unsigned __int8 **)v17;
LABEL_29:
  if ( v109 )
    sub_B91220((__int64)&v109, v109);
  return v21;
}
