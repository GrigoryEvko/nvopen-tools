// Function: sub_1B21730
// Address: 0x1b21730
//
__int64 __fastcall sub_1B21730(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // r13
  _BYTE *v19; // rsi
  void **v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  unsigned __int64 v24; // r14
  __int64 v25; // rax
  unsigned __int64 v26; // rsi
  unsigned __int64 v27; // rdx
  __int64 v28; // rax
  char v29; // di
  int v30; // r9d
  __int64 v31; // rcx
  __int64 v32; // r8
  unsigned __int64 v33; // r15
  __int64 v34; // rax
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  char v38; // si
  unsigned __int64 v39; // r14
  unsigned __int64 v40; // rax
  __int64 v41; // r15
  __int64 *v42; // rax
  char v43; // dl
  __int64 v44; // r13
  __int64 *v45; // rax
  __int64 *v46; // rcx
  __int64 *v47; // rsi
  unsigned __int64 v48; // rdx
  char v49; // si
  char v50; // cl
  __int64 *v51; // r15
  unsigned int v52; // r13d
  __int64 v53; // r15
  __int64 *v54; // r14
  unsigned int v55; // r12d
  __int64 v56; // rax
  _QWORD *v57; // r14
  _QWORD *v58; // r13
  unsigned __int64 v59; // rdi
  _QWORD *v60; // r13
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rax
  _QWORD *v66; // r14
  _QWORD *v67; // r13
  __int64 v68; // rsi
  __int64 *v69; // [rsp+10h] [rbp-410h]
  __int64 v70; // [rsp+28h] [rbp-3F8h]
  __int64 v71; // [rsp+30h] [rbp-3F0h]
  __int64 *v72; // [rsp+38h] [rbp-3E8h]
  _QWORD *v73; // [rsp+38h] [rbp-3E8h]
  __int64 v74; // [rsp+40h] [rbp-3E0h]
  __int64 v75; // [rsp+48h] [rbp-3D8h]
  __int64 *v76; // [rsp+58h] [rbp-3C8h]
  __int64 *v77; // [rsp+58h] [rbp-3C8h]
  __int64 v78; // [rsp+68h] [rbp-3B8h] BYREF
  __int64 v79; // [rsp+70h] [rbp-3B0h] BYREF
  char v80; // [rsp+80h] [rbp-3A0h]
  __int64 *v81; // [rsp+90h] [rbp-390h] BYREF
  __int64 v82; // [rsp+98h] [rbp-388h]
  _BYTE v83[64]; // [rsp+A0h] [rbp-380h] BYREF
  char *v84; // [rsp+E0h] [rbp-340h] BYREF
  __int64 v85; // [rsp+E8h] [rbp-338h] BYREF
  unsigned __int64 v86; // [rsp+F0h] [rbp-330h]
  __int64 v87; // [rsp+F8h] [rbp-328h]
  __int64 v88; // [rsp+100h] [rbp-320h]
  _BYTE v89[64]; // [rsp+108h] [rbp-318h] BYREF
  unsigned __int64 v90; // [rsp+148h] [rbp-2D8h] BYREF
  unsigned __int64 v91; // [rsp+150h] [rbp-2D0h]
  unsigned __int64 v92; // [rsp+158h] [rbp-2C8h]
  unsigned __int64 *v93; // [rsp+160h] [rbp-2C0h] BYREF
  __int64 v94; // [rsp+168h] [rbp-2B8h] BYREF
  unsigned __int64 v95; // [rsp+170h] [rbp-2B0h] BYREF
  __int64 v96; // [rsp+178h] [rbp-2A8h]
  __int64 v97; // [rsp+180h] [rbp-2A0h]
  _BYTE v98[64]; // [rsp+188h] [rbp-298h] BYREF
  unsigned __int64 v99; // [rsp+1C8h] [rbp-258h]
  unsigned __int64 v100; // [rsp+1D0h] [rbp-250h]
  unsigned __int64 v101; // [rsp+1D8h] [rbp-248h]
  __int64 v102[2]; // [rsp+1E0h] [rbp-240h] BYREF
  unsigned __int64 v103; // [rsp+1F0h] [rbp-230h]
  _QWORD *v104; // [rsp+1F8h] [rbp-228h]
  unsigned int v105; // [rsp+208h] [rbp-218h]
  _QWORD *v106; // [rsp+218h] [rbp-208h]
  unsigned int v107; // [rsp+228h] [rbp-1F8h]
  char v108; // [rsp+230h] [rbp-1F0h]
  _QWORD *v109; // [rsp+240h] [rbp-1E0h]
  __int64 v110; // [rsp+248h] [rbp-1D8h]
  _QWORD v111[2]; // [rsp+250h] [rbp-1D0h] BYREF
  char v112[8]; // [rsp+260h] [rbp-1C0h] BYREF
  __int64 v113; // [rsp+268h] [rbp-1B8h]
  unsigned __int64 v114; // [rsp+270h] [rbp-1B0h]
  void *v115; // [rsp+290h] [rbp-190h]
  __int64 *v116; // [rsp+2B8h] [rbp-168h]
  __int64 v117; // [rsp+2C8h] [rbp-158h] BYREF
  __int64 v118; // [rsp+2D0h] [rbp-150h]
  __int64 v119; // [rsp+2D8h] [rbp-148h]
  _QWORD *v120; // [rsp+350h] [rbp-D0h]
  unsigned int v121; // [rsp+360h] [rbp-C0h]
  __int64 v122; // [rsp+370h] [rbp-B0h]
  __int64 v123; // [rsp+390h] [rbp-90h]
  __int64 v124; // [rsp+3B0h] [rbp-70h]

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
    goto LABEL_148;
  while ( *(_UNKNOWN **)v2 != &unk_4F9920C )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_148;
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F9920C);
  v5 = *(__int64 **)(a1 + 8);
  v6 = v4;
  v71 = v4 + 160;
  v7 = *v5;
  v8 = v5[1];
  if ( v7 == v8 )
    goto LABEL_148;
  while ( *(_UNKNOWN **)v7 != &unk_5051F8C )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_148;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_5051F8C);
  v10 = *(__int64 **)(a1 + 8);
  v70 = v9;
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
    goto LABEL_148;
  while ( *(_UNKNOWN **)v11 != &unk_4F9E06C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_148;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9E06C);
  v14 = *(__int64 **)(a1 + 8);
  v75 = v13 + 160;
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_148:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4F9A488 )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_148;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4F9A488);
  v18 = *(__int64 **)(v6 + 192);
  v74 = *(_QWORD *)(v17 + 160);
  v81 = (__int64 *)v83;
  v82 = 0x800000000LL;
  v72 = *(__int64 **)(v6 + 200);
  if ( v18 == v72 )
    return 0;
  v76 = v18;
  do
  {
    v78 = *v76;
    sub_1B1EDD0(v102, &v78);
    v19 = v89;
    v20 = (void **)&v84;
    sub_16CCCB0(&v84, (__int64)v89, (__int64)v102);
    v22 = v111[0];
    v23 = v110;
    v90 = 0;
    v91 = 0;
    v92 = 0;
    v24 = v111[0] - v110;
    if ( v111[0] == v110 )
    {
      v24 = 0;
      v26 = 0;
    }
    else
    {
      if ( v24 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_147;
      v25 = sub_22077B0(v111[0] - v110);
      v22 = v111[0];
      v23 = v110;
      v26 = v25;
    }
    v90 = v26;
    v91 = v26;
    v92 = v26 + v24;
    if ( v22 != v23 )
    {
      v27 = v26;
      v28 = v23;
      do
      {
        if ( v27 )
        {
          *(_QWORD *)v27 = *(_QWORD *)v28;
          v29 = *(_BYTE *)(v28 + 16);
          *(_BYTE *)(v27 + 16) = v29;
          if ( v29 )
            *(_QWORD *)(v27 + 8) = *(_QWORD *)(v28 + 8);
        }
        v28 += 24;
        v27 += 24LL;
      }
      while ( v22 != v28 );
      v26 += 8 * ((unsigned __int64)(v22 - 24 - v23) >> 3) + 24;
    }
    v91 = v26;
    v19 = v98;
    v20 = (void **)&v93;
    sub_16CCCB0(&v93, (__int64)v98, (__int64)v112);
    v31 = v118;
    v32 = v117;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v33 = v118 - v117;
    if ( v118 == v117 )
    {
      v35 = 0;
    }
    else
    {
      if ( v33 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_147:
        sub_4261EA(v20, v19, v21);
      v34 = sub_22077B0(v118 - v117);
      v31 = v118;
      v32 = v117;
      v35 = v34;
    }
    v99 = v35;
    v36 = v35;
    v100 = v35;
    v101 = v35 + v33;
    if ( v31 != v32 )
    {
      v37 = v32;
      do
      {
        if ( v36 )
        {
          *(_QWORD *)v36 = *(_QWORD *)v37;
          v38 = *(_BYTE *)(v37 + 16);
          *(_BYTE *)(v36 + 16) = v38;
          if ( v38 )
            *(_QWORD *)(v36 + 8) = *(_QWORD *)(v37 + 8);
        }
        v37 += 24;
        v36 += 24LL;
      }
      while ( v37 != v31 );
      v36 = v35 + 8 * ((unsigned __int64)(v37 - 24 - v32) >> 3) + 24;
    }
    v39 = v91;
    v40 = v90;
    v100 = v36;
    if ( v91 - v90 == v36 - v35 )
      goto LABEL_56;
    do
    {
LABEL_40:
      v41 = *(_QWORD *)(v39 - 24);
      if ( *(_QWORD *)(v41 + 16) == *(_QWORD *)(v41 + 8) )
      {
        v64 = (unsigned int)v82;
        if ( (unsigned int)v82 >= HIDWORD(v82) )
        {
          sub_16CD150((__int64)&v81, v83, 0, 8, v32, v30);
          v64 = (unsigned int)v82;
        }
        v81[v64] = v41;
        v39 = v91;
        LODWORD(v82) = v82 + 1;
LABEL_117:
        v41 = *(_QWORD *)(v39 - 24);
      }
      if ( !*(_BYTE *)(v39 - 8) )
      {
        v42 = *(__int64 **)(v41 + 8);
        *(_BYTE *)(v39 - 8) = 1;
        *(_QWORD *)(v39 - 16) = v42;
        goto LABEL_45;
      }
      while ( 1 )
      {
        v42 = *(__int64 **)(v39 - 16);
LABEL_45:
        if ( *(__int64 **)(v41 + 16) == v42 )
          break;
        *(_QWORD *)(v39 - 16) = v42 + 1;
        v44 = *v42;
        v45 = (__int64 *)v85;
        if ( v86 != v85 )
          goto LABEL_43;
        v46 = (__int64 *)(v85 + 8LL * HIDWORD(v87));
        if ( (__int64 *)v85 == v46 )
        {
LABEL_114:
          if ( HIDWORD(v87) < (unsigned int)v87 )
          {
            ++HIDWORD(v87);
            *v46 = v44;
            ++v84;
LABEL_54:
            v79 = v44;
            v80 = 0;
            sub_197E9F0(&v90, (__int64)&v79);
            v39 = v91;
            v40 = v90;
            goto LABEL_55;
          }
LABEL_43:
          sub_16CCBA0((__int64)&v84, v44);
          if ( v43 )
            goto LABEL_54;
        }
        else
        {
          v47 = 0;
          while ( v44 != *v45 )
          {
            if ( *v45 == -2 )
            {
              v47 = v45;
              if ( v45 + 1 == v46 )
                goto LABEL_53;
              ++v45;
            }
            else if ( v46 == ++v45 )
            {
              if ( !v47 )
                goto LABEL_114;
LABEL_53:
              *v47 = v44;
              LODWORD(v88) = v88 - 1;
              ++v84;
              goto LABEL_54;
            }
          }
        }
      }
      v91 -= 24LL;
      v40 = v90;
      v39 = v91;
      if ( v91 != v90 )
        goto LABEL_117;
LABEL_55:
      v35 = v99;
    }
    while ( v39 - v40 != v100 - v99 );
LABEL_56:
    if ( v39 != v40 )
    {
      v48 = v35;
      while ( *(_QWORD *)v40 == *(_QWORD *)v48 )
      {
        v49 = *(_BYTE *)(v40 + 16);
        v50 = *(_BYTE *)(v48 + 16);
        if ( v49 && v50 )
        {
          if ( *(_QWORD *)(v40 + 8) != *(_QWORD *)(v48 + 8) )
            goto LABEL_40;
        }
        else if ( v49 != v50 )
        {
          goto LABEL_40;
        }
        v40 += 24LL;
        v48 += 24LL;
        if ( v40 == v39 )
          goto LABEL_63;
      }
      goto LABEL_40;
    }
LABEL_63:
    if ( v35 )
      j_j___libc_free_0(v35, v101 - v35);
    if ( v95 != v94 )
      _libc_free(v95);
    if ( v90 )
      j_j___libc_free_0(v90, v92 - v90);
    if ( v86 != v85 )
      _libc_free(v86);
    if ( v117 )
      j_j___libc_free_0(v117, v119 - v117);
    if ( v114 != v113 )
      _libc_free(v114);
    if ( v110 )
      j_j___libc_free_0(v110, v111[1] - v110);
    if ( v103 != v102[1] )
      _libc_free(v103);
    ++v76;
  }
  while ( v72 != v76 );
  v51 = &v81[(unsigned int)v82];
  if ( v81 == v51 )
  {
    v52 = 0;
  }
  else
  {
    v77 = v81;
    v52 = 0;
    v69 = &v81[(unsigned int)v82];
    do
    {
      v53 = *v77;
      v54 = (__int64 *)sub_38694E0(v70);
      v55 = sub_13FCBF0(v53);
      if ( (_BYTE)v55 )
      {
        if ( *(_DWORD *)(v54[1] + 280) || (v56 = sub_1458800(*v54), !sub_1452CB0(v56)) )
        {
          sub_1B1E040((__int64)v102, v54, v53, v71, v75, v74, 1);
          sub_1B17630((__int64)&v93, v102[0]);
          sub_1B1F0F0((__int64)v102, (__int64)&v93);
          if ( v93 != &v95 )
            _libc_free((unsigned __int64)v93);
          sub_1B216C0((__int64)v102);
          j___libc_free_0(v124);
          j___libc_free_0(v123);
          j___libc_free_0(v122);
          v115 = &unk_49EC708;
          if ( v121 )
          {
            v57 = v120;
            v58 = &v120[7 * v121];
            do
            {
              if ( *v57 != -16 && *v57 != -8 )
              {
                v59 = v57[1];
                if ( (_QWORD *)v59 != v57 + 3 )
                  _libc_free(v59);
              }
              v57 += 7;
            }
            while ( v58 != v57 );
          }
          j___libc_free_0(v120);
          if ( v116 != &v117 )
            _libc_free((unsigned __int64)v116);
          if ( v109 != v111 )
            _libc_free((unsigned __int64)v109);
          if ( v108 )
          {
            if ( v107 )
            {
              v66 = v106;
              v67 = &v106[2 * v107];
              do
              {
                if ( *v66 != -8 && *v66 != -4 )
                {
                  v68 = v66[1];
                  if ( v68 )
                    sub_161E7C0((__int64)(v66 + 1), v68);
                }
                v66 += 2;
              }
              while ( v67 != v66 );
            }
            j___libc_free_0(v106);
          }
          if ( v105 )
          {
            v60 = v104;
            v85 = 2;
            v86 = 0;
            v87 = -8;
            v61 = -8;
            v84 = (char *)&unk_49E6B50;
            v88 = 0;
            v94 = 2;
            v95 = 0;
            v96 = -16;
            v93 = (unsigned __int64 *)&unk_49E6B50;
            v97 = 0;
            v73 = &v104[8 * (unsigned __int64)v105];
            while ( 1 )
            {
              v62 = v60[3];
              if ( v62 != v61 )
              {
                v61 = v96;
                if ( v62 != v96 )
                {
                  v63 = v60[7];
                  if ( v63 == 0 || v63 == -8 || v63 == -16 )
                  {
                    v61 = v60[3];
                  }
                  else
                  {
                    sub_1649B30(v60 + 5);
                    v61 = v60[3];
                  }
                }
              }
              *v60 = &unk_49EE2B0;
              if ( v61 != 0 && v61 != -8 && v61 != -16 )
                sub_1649B30(v60 + 1);
              v60 += 8;
              if ( v73 == v60 )
                break;
              v61 = v87;
            }
            v93 = (unsigned __int64 *)&unk_49EE2B0;
            if ( v96 != 0 && v96 != -8 && v96 != -16 )
              sub_1649B30(&v94);
            v84 = (char *)&unk_49EE2B0;
            if ( v87 != 0 && v87 != -8 && v87 != -16 )
              sub_1649B30(&v85);
          }
          v52 = v55;
          j___libc_free_0(v104);
        }
      }
      ++v77;
    }
    while ( v69 != v77 );
    v51 = v81;
  }
  if ( v51 != (__int64 *)v83 )
    _libc_free((unsigned __int64)v51);
  return v52;
}
