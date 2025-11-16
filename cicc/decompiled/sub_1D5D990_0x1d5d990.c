// Function: sub_1D5D990
// Address: 0x1d5d990
//
__int64 __fastcall sub_1D5D990(
        _QWORD *a1,
        _QWORD *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r12d
  __int64 v11; // r15
  int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rbx
  _QWORD *v18; // rax
  double v19; // xmm4_8
  double v20; // xmm5_8
  __int64 v21; // r13
  int v22; // esi
  __int64 v23; // rax
  __int64 *v24; // r14
  __int64 *v25; // r15
  char v26; // al
  __int64 v27; // rdx
  unsigned int v28; // eax
  _BYTE *v29; // r13
  __int64 (*v30)(); // rbx
  unsigned int v32; // eax
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // r14
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 ***v38; // rax
  unsigned __int8 v39; // al
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // r12
  _QWORD **v43; // rax
  __int64 v44; // r13
  _QWORD **v45; // r8
  _QWORD **i; // r15
  _QWORD *v47; // r13
  _QWORD *v48; // rax
  unsigned int v49; // eax
  int v50; // r14d
  char v51; // cl
  __int64 **v52; // rax
  __int64 v53; // rax
  __int64 v54; // rbx
  unsigned int v55; // r11d
  __int64 v56; // rax
  int v57; // r8d
  int v58; // r9d
  __int64 v59; // rsi
  int v60; // ebx
  __int64 v61; // r12
  __int64 v62; // rbx
  unsigned int v63; // eax
  __int64 v64; // rdx
  __int64 *v65; // rax
  __int64 v66; // rcx
  unsigned __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 *v69; // rax
  __int64 v70; // rsi
  unsigned __int64 v71; // rcx
  __int64 v72; // rsi
  _QWORD *v73; // rsi
  __int64 v74; // rcx
  unsigned int v75; // esi
  __int64 *v76; // r10
  __int64 v77; // r10
  __int64 *v78; // r9
  __int64 v79; // rax
  __int64 v80; // r14
  unsigned int v81; // r15d
  __int64 (*v82)(); // rbx
  __int64 v83; // rax
  __int64 v84; // rdx
  int v85; // eax
  unsigned __int64 v86; // rbx
  unsigned __int64 v87; // r12
  __int64 *v88; // [rsp+8h] [rbp-128h]
  __int64 *v89; // [rsp+18h] [rbp-118h]
  __int64 v90; // [rsp+20h] [rbp-110h]
  _BYTE *v91; // [rsp+30h] [rbp-100h]
  int v92; // [rsp+3Ch] [rbp-F4h]
  unsigned int v93; // [rsp+3Ch] [rbp-F4h]
  _QWORD **v94; // [rsp+58h] [rbp-D8h]
  int v95; // [rsp+6Ch] [rbp-C4h] BYREF
  __int64 *v96; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v97; // [rsp+78h] [rbp-B8h]
  _BYTE v98[32]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v99; // [rsp+A0h] [rbp-90h]
  __int64 v100; // [rsp+A8h] [rbp-88h]
  __int64 v101; // [rsp+B0h] [rbp-80h]
  _QWORD *v102; // [rsp+B8h] [rbp-78h]
  _BYTE *v103; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v104; // [rsp+C8h] [rbp-68h]
  _BYTE v105[32]; // [rsp+D0h] [rbp-60h] BYREF
  unsigned int v106; // [rsp+F0h] [rbp-40h]
  _QWORD *v107; // [rsp+F8h] [rbp-38h]

  v10 = (unsigned __int8)byte_4FC2EA0;
  v95 = -1;
  if ( byte_4FC2EA0 )
    return 0;
  v11 = a1[22];
  if ( !v11 )
    return 0;
  if ( byte_4FC2DC0 )
  {
    v13 = -1;
    goto LABEL_5;
  }
  v30 = *(__int64 (**)())(*(_QWORD *)v11 + 248LL);
  if ( v30 == sub_1D5A3B0 )
    return 0;
  v40 = sub_13CF970((__int64)a2);
  if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, int *))v30)(
          v11,
          **(_QWORD **)v40,
          *(_QWORD *)(v40 + 24),
          &v95) )
    return 0;
  v13 = v95;
  v11 = a1[22];
LABEL_5:
  v14 = a1[24];
  v15 = a2[1];
  v102 = a2;
  v16 = a1[113];
  v17 = a2[5];
  v106 = v13;
  v101 = v14;
  v103 = v105;
  v99 = v16;
  v100 = v11;
  v104 = 0x400000000LL;
  v107 = 0;
  if ( !v15 )
    return v10;
  while ( 1 )
  {
    if ( *(_QWORD *)(v15 + 8) )
      goto LABEL_25;
    v18 = sub_1648700(v15);
    v21 = (__int64)v18;
    if ( v17 != v18[5] )
      goto LABEL_25;
    v22 = *((unsigned __int8 *)v18 + 16);
    if ( (_BYTE)v22 == 55 )
      break;
    if ( (unsigned int)(v22 - 35) > 0x11 )
      goto LABEL_25;
    v23 = 3LL * (*((_DWORD *)v18 + 5) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v21 + 23) & 0x40) != 0 )
    {
      v24 = *(__int64 **)(v21 - 8);
      v25 = &v24[v23];
    }
    else
    {
      v25 = (__int64 *)v21;
      v24 = (__int64 *)(v21 - v23 * 8);
    }
    if ( v24 != v25 )
    {
      while ( 1 )
      {
        v27 = *v24;
        if ( (_DWORD)v104 )
        {
          if ( v27 != *(_QWORD *)&v103[8 * (unsigned int)v104 - 8] )
            goto LABEL_15;
        }
        else if ( (_QWORD *)v27 != v102 )
        {
LABEL_15:
          v26 = *(_BYTE *)(v27 + 16);
          if ( (unsigned __int8)(v26 - 13) > 1u && v26 != 9 )
            goto LABEL_25;
          goto LABEL_17;
        }
        if ( (unsigned int)sub_1648720((__int64)v24) != 1 )
          goto LABEL_17;
        v28 = *(unsigned __int8 *)(v21 + 16) - 24;
        if ( v28 > 0x15 )
        {
          if ( *(_BYTE *)(v21 + 16) == 46 )
            goto LABEL_24;
LABEL_17:
          v24 += 3;
          if ( v25 == v24 )
            goto LABEL_33;
        }
        else
        {
          if ( v28 > 0x13 )
            goto LABEL_25;
          if ( *(_BYTE *)(v21 + 16) == 43 )
          {
LABEL_24:
            if ( !sub_15F24B0(v21) )
              goto LABEL_25;
            goto LABEL_17;
          }
          if ( v28 > 0x10 )
            goto LABEL_25;
          v24 += 3;
          if ( v25 == v24 )
          {
LABEL_33:
            v22 = *(unsigned __int8 *)(v21 + 16);
            break;
          }
        }
      }
    }
    v32 = sub_1F43D70(v100, (unsigned int)(v22 - 24));
    v35 = v32;
    if ( !v32 )
      goto LABEL_25;
    if ( !byte_4FC2DC0 )
    {
      v37 = v100;
      v38 = (__int64 ***)sub_13CF970((__int64)v102);
      v39 = sub_1D5D7E0(v99, **v38, 1u);
      if ( v39 != 1 && (!v39 || !*(_QWORD *)(v37 + 8LL * v39 + 120)) )
        goto LABEL_46;
      if ( (unsigned int)v35 <= 0x102 && (*(_BYTE *)(v35 + 259LL * v39 + v37 + 2422) & 0xFB) != 0 )
        goto LABEL_46;
    }
    v36 = (unsigned int)v104;
    if ( (unsigned int)v104 >= HIDWORD(v104) )
    {
      sub_16CD150((__int64)&v103, v105, 0, 8, v33, v34);
      v36 = (unsigned int)v104;
    }
    *(_QWORD *)&v103[8 * v36] = v21;
    LODWORD(v104) = v104 + 1;
    v15 = *(_QWORD *)(v21 + 8);
    if ( !v15 )
      goto LABEL_25;
  }
  v107 = v18;
  v41 = (unsigned int)v104;
  if ( !(_DWORD)v104 )
  {
LABEL_25:
    v29 = v103;
    goto LABEL_26;
  }
  if ( byte_4FC2DC0 )
  {
LABEL_54:
    v29 = &v103[8 * v41];
    if ( v103 != v29 )
    {
      v89 = (__int64 *)v103;
      v88 = (__int64 *)&v103[8 * v41];
      do
      {
        v42 = *v89;
        sub_164D160(*v89, (__int64)v102, a3, a4, a5, a6, v19, v20, a9, a10);
        if ( (*((_BYTE *)v102 + 23) & 0x40) != 0 )
          v43 = (_QWORD **)*(v102 - 1);
        else
          v43 = (_QWORD **)&v102[-3 * (*((_DWORD *)v102 + 5) & 0xFFFFFFF)];
        *(_QWORD *)v42 = **v43;
        v44 = 3LL * (*(_DWORD *)(v42 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v42 + 23) & 0x40) != 0 )
        {
          v45 = *(_QWORD ***)(v42 - 8);
          v94 = &v45[v44];
        }
        else
        {
          v94 = (_QWORD **)v42;
          v45 = (_QWORD **)(v42 - v44 * 8);
        }
        for ( i = v45; v94 != i; i += 3 )
        {
          v47 = *i;
          v48 = v102;
          if ( *i == v102 )
          {
            if ( (*((_BYTE *)v47 + 23) & 0x40) != 0 )
              v78 = (__int64 *)*(v47 - 1);
            else
              v78 = &v47[-3 * (*((_DWORD *)v47 + 5) & 0xFFFFFFF)];
            v62 = *v78;
            goto LABEL_84;
          }
          if ( *((_BYTE *)v47 + 16) == 9 )
            goto LABEL_69;
          if ( (unsigned int)sub_1648720((__int64)i) == 1 )
          {
            v49 = *(unsigned __int8 *)(v42 + 16) - 24;
            if ( v49 > 0x15 )
            {
              if ( *(_BYTE *)(v42 + 16) != 46 )
                goto LABEL_106;
LABEL_67:
              if ( !sub_15F24B0(v42) )
                goto LABEL_68;
              goto LABEL_106;
            }
            if ( v49 > 0x13 )
              goto LABEL_68;
            if ( *(_BYTE *)(v42 + 16) == 43 )
              goto LABEL_67;
            if ( v49 > 0x10 )
            {
LABEL_68:
              v48 = v102;
LABEL_69:
              v50 = -1;
              v51 = 1;
              if ( (*((_BYTE *)v48 + 23) & 0x40) != 0 )
                goto LABEL_70;
LABEL_111:
              v52 = (__int64 **)&v48[-3 * (*((_DWORD *)v48 + 5) & 0xFFFFFFF)];
LABEL_71:
              v53 = **v52;
              v54 = *(_QWORD *)(v53 + 32);
              v55 = v54;
              if ( !v51 )
              {
                v92 = *(_QWORD *)(v53 + 32);
                v96 = (__int64 *)v98;
                v97 = 0x400000000LL;
                v56 = sub_1599EF0((__int64 **)*v47);
                v59 = (unsigned int)v97;
                if ( (_DWORD)v54 )
                {
                  v90 = v42;
                  v60 = 0;
                  v61 = v56;
                  do
                  {
                    while ( v50 == v60 )
                    {
                      if ( (unsigned int)v59 >= HIDWORD(v97) )
                      {
                        sub_16CD150((__int64)&v96, v98, 0, 8, v57, v58);
                        v59 = (unsigned int)v97;
                      }
                      ++v60;
                      v96[v59] = (__int64)v47;
                      v59 = (unsigned int)(v97 + 1);
                      LODWORD(v97) = v97 + 1;
                      if ( v92 == v60 )
                        goto LABEL_81;
                    }
                    if ( (unsigned int)v59 >= HIDWORD(v97) )
                    {
                      sub_16CD150((__int64)&v96, v98, 0, 8, v57, v58);
                      v59 = (unsigned int)v97;
                    }
                    ++v60;
                    v96[v59] = v61;
                    v59 = (unsigned int)(v97 + 1);
                    LODWORD(v97) = v97 + 1;
                  }
                  while ( v92 != v60 );
LABEL_81:
                  v42 = v90;
                }
                v62 = sub_15A01B0(v96, v59);
                if ( v96 != (__int64 *)v98 )
                  _libc_free((unsigned __int64)v96);
                goto LABEL_84;
              }
              goto LABEL_114;
            }
          }
LABEL_106:
          v48 = v102;
          if ( (*((_BYTE *)v102 + 23) & 0x40) != 0 )
          {
            v73 = (_QWORD *)*(v102 - 1);
            v74 = v73[3];
            if ( *(_BYTE *)(v74 + 16) == 13 )
              goto LABEL_108;
          }
          else
          {
            v73 = &v102[-3 * (*((_DWORD *)v102 + 5) & 0xFFFFFFF)];
            v74 = v73[3];
            if ( *(_BYTE *)(v74 + 16) == 13 )
            {
LABEL_108:
              v75 = *(_DWORD *)(v74 + 32);
              v76 = *(__int64 **)(v74 + 24);
              if ( v75 > 0x40 )
                v77 = *v76;
              else
                v77 = (__int64)((_QWORD)v76 << (64 - (unsigned __int8)v75)) >> (64 - (unsigned __int8)v75);
              v50 = v77;
              v51 = 0;
              if ( (*((_BYTE *)v102 + 23) & 0x40) == 0 )
                goto LABEL_111;
LABEL_70:
              v52 = (__int64 **)*(v48 - 1);
              goto LABEL_71;
            }
          }
          v55 = *(_DWORD *)(*(_QWORD *)*v73 + 32LL);
LABEL_114:
          v62 = sub_15A0390(v55, (__int64)v47);
LABEL_84:
          v63 = sub_1648720((__int64)i);
          if ( (*(_BYTE *)(v42 + 23) & 0x40) != 0 )
            v64 = *(_QWORD *)(v42 - 8);
          else
            v64 = v42 - 24LL * (*(_DWORD *)(v42 + 20) & 0xFFFFFFF);
          v65 = (__int64 *)(v64 + 24LL * v63);
          if ( *v65 )
          {
            v66 = v65[1];
            v67 = v65[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v67 = v66;
            if ( v66 )
              *(_QWORD *)(v66 + 16) = *(_QWORD *)(v66 + 16) & 3LL | v67;
          }
          *v65 = v62;
          if ( v62 )
          {
            v68 = *(_QWORD *)(v62 + 8);
            v65[1] = v68;
            if ( v68 )
              *(_QWORD *)(v68 + 16) = (unsigned __int64)(v65 + 1) | *(_QWORD *)(v68 + 16) & 3LL;
            v65[2] = (v62 + 8) | v65[2] & 3;
            *(_QWORD *)(v62 + 8) = v65;
          }
        }
        sub_15F2300(v102, v42);
        if ( (*((_BYTE *)v102 + 23) & 0x40) != 0 )
          v69 = (__int64 *)*(v102 - 1);
        else
          v69 = &v102[-3 * (*((_DWORD *)v102 + 5) & 0xFFFFFFF)];
        if ( *v69 )
        {
          v70 = v69[1];
          v71 = v69[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v71 = v70;
          if ( v70 )
            *(_QWORD *)(v70 + 16) = v71 | *(_QWORD *)(v70 + 16) & 3LL;
        }
        *v69 = v42;
        v72 = *(_QWORD *)(v42 + 8);
        v69[1] = v72;
        if ( v72 )
          *(_QWORD *)(v72 + 16) = (unsigned __int64)(v69 + 1) | *(_QWORD *)(v72 + 16) & 3LL;
        ++v89;
        v69[2] = (v42 + 8) | v69[2] & 3;
        *(_QWORD *)(v42 + 8) = v69;
      }
      while ( v88 != v89 );
      v29 = v103;
    }
    goto LABEL_103;
  }
  v79 = **(_QWORD **)(v21 - 24);
  if ( *(_BYTE *)(v79 + 8) == 16 )
    v79 = **(_QWORD **)(v79 + 16);
  v80 = v100;
  v81 = *(unsigned __int16 *)(v21 + 18);
  v93 = *(_DWORD *)(v79 + 8) >> 8;
  v82 = *(__int64 (**)())(*(_QWORD *)v100 + 448LL);
  v83 = sub_1D5D7E0(v99, **(__int64 ***)(v21 - 48), 0);
  if ( v82 == sub_1D12D60
    || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD))v82)(
          v80,
          v83,
          v84,
          v93,
          (unsigned int)(1 << (v81 >> 1) >> 1),
          0) )
  {
LABEL_46:
    v29 = v103;
    v10 = 0;
    goto LABEL_26;
  }
  v85 = sub_14A3470(v101);
  v29 = v103;
  v86 = v85;
  if ( v103 != &v103[8 * (unsigned int)v104] )
  {
    v91 = &v103[8 * (unsigned int)v104];
    v87 = v106;
    do
    {
      v29 += 8;
      v86 += (int)sub_14A3350(v101);
      v87 += (int)sub_14A3350(v101);
    }
    while ( v91 != v29 );
    v41 = (unsigned int)v104;
    if ( v86 > v87 )
      goto LABEL_54;
    goto LABEL_46;
  }
  v10 = 0;
  if ( v85 > (unsigned __int64)v106 )
LABEL_103:
    v10 = 1;
LABEL_26:
  if ( v29 != v105 )
    _libc_free((unsigned __int64)v29);
  return v10;
}
