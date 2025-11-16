// Function: sub_2BD62F0
// Address: 0x2bd62f0
//
__int64 __fastcall sub_2BD62F0(
        __int64 a1,
        unsigned __int8 *a2,
        unsigned __int8 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  unsigned __int8 *v8; // r12
  _QWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  _QWORD *v15; // rdx
  unsigned __int64 v16; // rax
  int v17; // eax
  __int64 v18; // rcx
  int v19; // edx
  unsigned int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // r12
  __int64 v23; // rdx
  int v24; // edi
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rdx
  __int64 *v28; // rbx
  __int64 v29; // rdx
  unsigned __int64 *v30; // r14
  int v31; // eax
  unsigned __int64 *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rax
  unsigned __int8 **v36; // r14
  unsigned __int8 **v37; // rbx
  unsigned __int8 *v38; // r14
  unsigned __int8 **v39; // rax
  unsigned __int8 v40; // r14
  _QWORD *v41; // rbx
  _QWORD *v42; // rax
  __int64 v43; // r9
  __int64 v44; // r8
  __int64 v45; // rcx
  _QWORD *v46; // r15
  __int64 v47; // rsi
  unsigned __int64 v48; // rdi
  unsigned __int64 *v49; // rbx
  unsigned __int64 *v50; // r15
  __int64 v51; // rbx
  unsigned __int64 *v52; // r15
  unsigned __int64 v53; // rdi
  unsigned __int64 *v54; // rbx
  unsigned __int64 *v55; // r12
  unsigned __int64 v56; // rdi
  int v57; // eax
  unsigned int v58; // eax
  __int64 v59; // rsi
  int v60; // edi
  unsigned __int8 **v61; // rax
  bool v62; // bl
  __int64 v63; // rax
  unsigned __int8 *v64; // rcx
  _QWORD *v65; // rbx
  _QWORD *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // r15
  char *v70; // rbx
  int v71; // eax
  int v72; // ebx
  int v73; // ebx
  unsigned __int8 *v78; // [rsp+48h] [rbp-778h]
  int v79; // [rsp+48h] [rbp-778h]
  bool v81; // [rsp+5Eh] [rbp-762h]
  unsigned __int8 v82; // [rsp+5Fh] [rbp-761h]
  int v83; // [rsp+64h] [rbp-75Ch] BYREF
  unsigned __int8 *v84; // [rsp+68h] [rbp-758h] BYREF
  __int64 v85; // [rsp+70h] [rbp-750h] BYREF
  __int64 v86; // [rsp+78h] [rbp-748h]
  unsigned __int64 v87; // [rsp+80h] [rbp-740h]
  __int64 v88; // [rsp+88h] [rbp-738h]
  __int64 v89; // [rsp+90h] [rbp-730h]
  unsigned __int64 *v90; // [rsp+98h] [rbp-728h]
  _QWORD *v91; // [rsp+A0h] [rbp-720h]
  __int64 v92; // [rsp+A8h] [rbp-718h]
  __int64 v93; // [rsp+B0h] [rbp-710h]
  _QWORD *v94; // [rsp+B8h] [rbp-708h]
  __int64 v95; // [rsp+C0h] [rbp-700h] BYREF
  unsigned __int8 **v96; // [rsp+C8h] [rbp-6F8h]
  __int64 v97; // [rsp+D0h] [rbp-6F0h]
  int v98; // [rsp+D8h] [rbp-6E8h]
  char v99; // [rsp+DCh] [rbp-6E4h]
  char v100; // [rsp+E0h] [rbp-6E0h] BYREF
  __int64 v101; // [rsp+120h] [rbp-6A0h] BYREF
  __int64 v102; // [rsp+128h] [rbp-698h]
  _QWORD v103[36]; // [rsp+130h] [rbp-690h] BYREF
  unsigned __int64 *v104; // [rsp+250h] [rbp-570h]
  __int64 v105; // [rsp+258h] [rbp-568h]
  _BYTE v106[64]; // [rsp+260h] [rbp-560h] BYREF
  __int64 v107; // [rsp+2A0h] [rbp-520h]
  __int64 v108; // [rsp+2A8h] [rbp-518h]
  _QWORD *v109; // [rsp+2B0h] [rbp-510h] BYREF
  unsigned int v110; // [rsp+2B8h] [rbp-508h]
  _QWORD v111[2]; // [rsp+730h] [rbp-90h] BYREF
  __int64 v112; // [rsp+740h] [rbp-80h]
  char v113; // [rsp+74Ch] [rbp-74h]
  _BYTE *v114; // [rsp+750h] [rbp-70h]
  __int64 v115; // [rsp+758h] [rbp-68h]
  _BYTE v116[96]; // [rsp+760h] [rbp-60h] BYREF

  v82 = qword_5010268;
  if ( !(_BYTE)qword_5010268 )
    return v82;
  v81 = 0;
  if ( a2 )
    v81 = (unsigned int)*a3 - 42 <= 0x11;
  v8 = a3;
  v82 = 0;
  if ( a4 != *((_QWORD *)a3 + 5) || *a3 == 84 )
    return v82;
  v86 = 8;
  v85 = sub_22077B0(0x40u);
  v10 = (_QWORD *)(v85 + 24);
  v11 = sub_22077B0(0x200u);
  v90 = (unsigned __int64 *)(v85 + 24);
  v14 = v11 + 512;
  *(_QWORD *)(v85 + 24) = v11;
  v88 = v11;
  v89 = v11 + 512;
  v94 = v10;
  v92 = v11;
  v93 = v11 + 512;
  v87 = v11;
  v91 = (_QWORD *)v11;
  if ( v81 )
  {
    if ( (unsigned __int8)sub_2B41310(a3) && (unsigned int)sub_2B27770((__int64)a3) )
    {
      v61 = (unsigned __int8 **)sub_986520((__int64)a3);
      if ( *a3 == 86 && (unsigned __int8)(**v61 - 82) <= 1u )
      {
        v73 = sub_2B27770((__int64)a3);
        if ( (unsigned int)(v73 - 6) > 3 )
          v61 = (unsigned __int8 **)(((unsigned int)(v73 - 12) < 4 ? 0x20 : 0) + sub_986520((__int64)a3));
        else
          v61 = (unsigned __int8 **)(sub_986520((__int64)a3) + 32);
      }
      v8 = *v61;
      v62 = sub_2B27F10((__int64)a3);
      v63 = sub_986520((__int64)a3);
      v14 = v93;
      v12 = v63;
      v64 = *(unsigned __int8 **)(32LL * v62 + v63 + 32);
      v11 = (__int64)v91;
      if ( v8 == a2 )
      {
        if ( *v64 < 0x1Du )
          v64 = a3;
        v8 = v64;
      }
      else if ( v64 == a2 )
      {
        if ( *v8 <= 0x1Cu )
          v8 = a3;
      }
      else
      {
        v8 = a3;
      }
    }
    else
    {
      v11 = (__int64)v91;
      v14 = v93;
    }
  }
  if ( v11 == v14 - 16 )
  {
    v65 = v94;
    if ( ((__int64)(v89 - v87) >> 4) + 32 * (v94 - v90 - 1) + ((v11 - v92) >> 4) == 0x7FFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    if ( (unsigned __int64)(v86 - (((__int64)v94 - v85) >> 3)) <= 1 )
    {
      sub_2B49870((unsigned __int64 *)&v85, 1u, 0);
      v65 = v94;
    }
    v65[1] = sub_22077B0(0x200u);
    v66 = v91;
    if ( v91 )
    {
      *v91 = v8;
      *((_DWORD *)v66 + 2) = 0;
    }
    v15 = (_QWORD *)*++v94;
    v67 = *v94 + 512LL;
    v92 = (__int64)v15;
    v93 = v67;
    v91 = v15;
  }
  else
  {
    if ( v11 )
    {
      *(_QWORD *)v11 = v8;
      *(_DWORD *)(v11 + 8) = 0;
      v11 = (__int64)v91;
    }
    v15 = (_QWORD *)(v11 + 16);
    v91 = (_QWORD *)(v11 + 16);
  }
  v95 = 0;
  v96 = (unsigned __int8 **)&v100;
  v16 = v87;
  v97 = 8;
  v98 = 0;
  v99 = 1;
  if ( (_QWORD *)v87 == v15 )
  {
    v82 = 0;
    goto LABEL_101;
  }
  v82 = 0;
  while ( 1 )
  {
    v22 = *(_QWORD *)v16;
    v83 = *(_DWORD *)(v16 + 8);
    if ( v16 == v89 - 16 )
    {
      j_j___libc_free_0(v88);
      v23 = *++v90 + 512;
      v88 = *v90;
      v89 = v23;
      v87 = v88;
    }
    else
    {
      v87 += 16LL;
    }
    v17 = *(_DWORD *)(a5 + 2000);
    v18 = *(_QWORD *)(a5 + 1984);
    if ( v17 )
    {
      v19 = v17 - 1;
      v20 = (v17 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v21 = *(_QWORD *)(v18 + 8LL * v20);
      if ( v22 == v21 )
        goto LABEL_17;
      v24 = 1;
      while ( v21 != -4096 )
      {
        v12 = (unsigned int)(v24 + 1);
        v20 = v19 & (v24 + v20);
        v21 = *(_QWORD *)(v18 + 8LL * v20);
        if ( v22 == v21 )
          goto LABEL_17;
        ++v24;
      }
    }
    if ( *(_BYTE *)(a5 + 2036) )
      break;
    if ( sub_C8CA60(a5 + 2008, v22) )
      goto LABEL_27;
LABEL_62:
    v40 = sub_2B41310((unsigned __int8 *)v22);
    if ( !v40 )
      goto LABEL_27;
    v41 = v111;
    v107 = 0;
    v101 = (__int64)v103;
    v102 = 0x200000000LL;
    v104 = (unsigned __int64 *)v106;
    v105 = 0x100000000LL;
    v42 = &v109;
    v108 = 1;
    do
    {
      *v42 = -4096;
      v42 += 9;
    }
    while ( v42 != v111 );
    v113 = 0;
    v114 = v116;
    v115 = 0x300000000LL;
    v111[0] = 6;
    v43 = *(_QWORD *)(a1 + 16);
    v44 = *(_QWORD *)(a1 + 64);
    v111[1] = 0;
    v45 = *(_QWORD *)a1;
    v112 = 0;
    v78 = 0;
    if ( (unsigned __int8)sub_2BC2A20((__int64)&v101, a5, v22, v45, v44, v43) )
      v78 = sub_2BD1C50(
              (__int64)&v101,
              (__int64 **)a5,
              *(_QWORD *)(a1 + 64),
              *(__int64 **)(a1 + 8),
              *(__int64 **)(a1 + 16),
              *(_QWORD *)(a1 + 48),
              a7);
    if ( v114 != v116 )
      _libc_free((unsigned __int64)v114);
    LOBYTE(v27) = v112 != -4096;
    if ( ((unsigned __int8)v27 & (v112 != 0)) != 0 && v112 != -8192 )
      sub_BD60C0(v111);
    if ( (v108 & 1) != 0 )
    {
      v46 = &v109;
    }
    else
    {
      v27 = v110;
      v46 = v109;
      v47 = 9LL * v110;
      if ( !v110 )
        goto LABEL_112;
      v41 = &v109[v47];
      if ( v109 == &v109[v47] )
        goto LABEL_112;
    }
    do
    {
      if ( *v46 != -8192 && *v46 != -4096 )
      {
        v48 = v46[1];
        if ( (_QWORD *)v48 != v46 + 3 )
          _libc_free(v48);
      }
      v46 += 9;
    }
    while ( v46 != v41 );
    if ( (v108 & 1) != 0 )
      goto LABEL_81;
    v46 = v109;
    v47 = 9LL * v110;
LABEL_112:
    sub_C7D6A0((__int64)v46, v47 * 8, 8);
LABEL_81:
    v49 = v104;
    v50 = &v104[8 * (unsigned __int64)(unsigned int)v105];
    if ( v104 != v50 )
    {
      do
      {
        v50 -= 8;
        if ( (unsigned __int64 *)*v50 != v50 + 2 )
          _libc_free(*v50);
      }
      while ( v49 != v50 );
      v50 = v104;
    }
    if ( v50 != (unsigned __int64 *)v106 )
      _libc_free((unsigned __int64)v50);
    v51 = v101;
    v52 = (unsigned __int64 *)(v101 + 144LL * (unsigned int)v102);
    if ( (unsigned __int64 *)v101 != v52 )
    {
      do
      {
        v52 -= 18;
        if ( (unsigned __int64 *)*v52 != v52 + 2 )
          _libc_free(*v52);
      }
      while ( (unsigned __int64 *)v51 != v52 );
      v52 = (unsigned __int64 *)v101;
    }
    if ( v52 != v103 )
      _libc_free((unsigned __int64)v52);
    if ( !v78 )
      goto LABEL_27;
    if ( *v78 <= 0x1Cu )
    {
      v57 = *(_DWORD *)(a5 + 2000);
      v34 = *(_QWORD *)(a5 + 1984);
      if ( !v57 )
        goto LABEL_110;
      v27 = (unsigned int)(v57 - 1);
      v82 = v40;
      v58 = v27 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v59 = *(_QWORD *)(v34 + 8LL * v58);
      if ( v59 != v22 )
      {
        v60 = 1;
        while ( v59 != -4096 )
        {
          v12 = (unsigned int)(v60 + 1);
          v58 = v27 & (v60 + v58);
          v59 = *(_QWORD *)(v34 + 8LL * v58);
          if ( v22 == v59 )
            goto LABEL_98;
          ++v60;
        }
LABEL_110:
        v82 = v40;
LABEL_43:
        if ( ++v83 < (unsigned int)qword_500FC48 )
        {
          v35 = 4LL * (*(_DWORD *)(v22 + 4) & 0x7FFFFFF);
          if ( (*(_BYTE *)(v22 + 7) & 0x40) != 0 )
          {
            v36 = *(unsigned __int8 ***)(v22 - 8);
            v22 = (__int64)&v36[v35];
          }
          else
          {
            v36 = (unsigned __int8 **)(v22 - v35 * 8);
          }
          if ( (unsigned __int8 **)v22 != v36 )
          {
            v37 = v36;
            v38 = *v36;
            if ( v99 )
            {
LABEL_48:
              v39 = v96;
              v27 = (__int64)&v96[HIDWORD(v97)];
              if ( v96 != (unsigned __int8 **)v27 )
              {
                do
                {
                  if ( v38 == *v39 )
                    goto LABEL_52;
                  ++v39;
                }
                while ( (unsigned __int8 **)v27 != v39 );
              }
              if ( HIDWORD(v97) >= (unsigned int)v97 )
                goto LABEL_54;
              ++HIDWORD(v97);
              *(_QWORD *)v27 = v38;
              ++v95;
            }
            else
            {
LABEL_54:
              while ( 1 )
              {
                sub_C8CC70((__int64)&v95, (__int64)v38, v27, v34, v12, v13);
                if ( (_BYTE)v27 )
                  break;
LABEL_52:
                v37 += 4;
                if ( (unsigned __int8 **)v22 == v37 )
                  goto LABEL_17;
                v38 = *v37;
                if ( v99 )
                  goto LABEL_48;
              }
            }
            if ( *v38 > 0x1Cu )
            {
              v84 = v38;
              if ( (unsigned __int8)(*v38 - 82) > 0xCu || (v27 = 4615, !_bittest64(&v27, (unsigned int)*v38 - 82)) )
              {
                v101 = (__int64)v38;
                if ( !sub_2B4B3F0(a5 + 1976, &v101) && a4 == *((_QWORD *)v38 + 5) )
                  sub_2B499F0((unsigned __int64 *)&v85, &v84, &v83);
              }
            }
            goto LABEL_52;
          }
        }
      }
LABEL_17:
      v16 = v87;
      if ( (_QWORD *)v87 == v91 )
        goto LABEL_99;
    }
    else
    {
      v101 = (__int64)v78;
      sub_2B499F0((unsigned __int64 *)&v85, &v101, &v83);
LABEL_98:
      v82 = v40;
      v16 = v87;
      if ( (_QWORD *)v87 == v91 )
        goto LABEL_99;
    }
  }
  v25 = *(_QWORD **)(a5 + 2016);
  v26 = &v25[*(unsigned int *)(a5 + 2028)];
  if ( v25 == v26 )
    goto LABEL_62;
  while ( v22 != *v25 )
  {
    if ( v26 == ++v25 )
      goto LABEL_62;
  }
LABEL_27:
  if ( (unsigned __int8 *)v22 != a3 || !v81 )
  {
    v27 = v22;
LABEL_30:
    if ( (unsigned __int8)(*(_BYTE *)v27 - 82) > 0xCu
      || (v34 = 4611, !_bittest64(&v34, (unsigned int)*(unsigned __int8 *)v27 - 82)) )
    {
      v101 = 6;
      v102 = 0;
      v103[0] = v27;
      if ( v27 != -8192 && v27 != -4096 )
        sub_BD73F0((__int64)&v101);
      v28 = &v101;
      v29 = *(unsigned int *)(a6 + 8);
      v30 = *(unsigned __int64 **)a6;
      v13 = v29 + 1;
      v31 = *(_DWORD *)(a6 + 8);
      if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
      {
        v69 = a6 + 16;
        if ( v30 > (unsigned __int64 *)&v101 || &v101 >= (__int64 *)&v30[3 * v29] )
        {
          v30 = (unsigned __int64 *)sub_C8D7D0(a6, a6 + 16, v29 + 1, 0x18u, (unsigned __int64 *)&v84, v13);
          sub_F17F80(a6, v30);
          v72 = (int)v84;
          if ( v69 != *(_QWORD *)a6 )
            _libc_free(*(_QWORD *)a6);
          v29 = *(unsigned int *)(a6 + 8);
          *(_DWORD *)(a6 + 12) = v72;
          *(_QWORD *)a6 = v30;
          v28 = &v101;
          v31 = v29;
        }
        else
        {
          v70 = (char *)((char *)&v101 - (char *)v30);
          v30 = (unsigned __int64 *)sub_C8D7D0(a6, a6 + 16, v29 + 1, 0x18u, (unsigned __int64 *)&v84, v13);
          sub_F17F80(a6, v30);
          v71 = (int)v84;
          if ( *(_QWORD *)a6 == v69 )
          {
            *(_QWORD *)a6 = v30;
            *(_DWORD *)(a6 + 12) = v71;
          }
          else
          {
            v79 = (int)v84;
            _libc_free(*(_QWORD *)a6);
            *(_QWORD *)a6 = v30;
            *(_DWORD *)(a6 + 12) = v79;
          }
          v28 = (__int64 *)&v70[(_QWORD)v30];
          v29 = *(unsigned int *)(a6 + 8);
          v31 = *(_DWORD *)(a6 + 8);
        }
      }
      v27 = 3 * v29;
      v32 = &v30[v27];
      if ( v32 )
      {
        *v32 = 6;
        v33 = v28[2];
        v32[1] = 0;
        v32[2] = v33;
        if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
          sub_BD6050(v32, *v28 & 0xFFFFFFFFFFFFFFF8LL);
        v31 = *(_DWORD *)(a6 + 8);
      }
      v34 = a6;
      *(_DWORD *)(a6 + 8) = v31 + 1;
      LOBYTE(v34) = v103[0] != -4096;
      LOBYTE(v27) = v103[0] != 0;
      if ( ((v103[0] != 0) & (unsigned __int8)v34) != 0 && v103[0] != -8192 )
        sub_BD60C0(&v101);
    }
    goto LABEL_43;
  }
  v84 = 0;
  v101 = 0;
  if ( (unsigned int)*a3 - 42 <= 0x11
    && (v27 = *((_QWORD *)a3 - 8)) != 0
    && (v68 = *((_QWORD *)a3 - 4), v84 = (unsigned __int8 *)*((_QWORD *)a3 - 8), v68) )
  {
    v101 = v68;
  }
  else
  {
    if ( !(unsigned __int8)sub_2B40F30((__int64)a3, &v84, &v101) )
      goto LABEL_99;
    v27 = (__int64)v84;
  }
  if ( a2 == (unsigned __int8 *)v27 )
    v27 = v101;
  if ( *(_BYTE *)v27 > 0x1Cu )
    goto LABEL_30;
LABEL_99:
  if ( !v99 )
    _libc_free((unsigned __int64)v96);
LABEL_101:
  v53 = v85;
  if ( v85 )
  {
    v54 = v90;
    v55 = v94 + 1;
    if ( v94 + 1 > v90 )
    {
      do
      {
        v56 = *v54++;
        j_j___libc_free_0(v56);
      }
      while ( v55 > v54 );
      v53 = v85;
    }
    j_j___libc_free_0(v53);
  }
  return v82;
}
