// Function: sub_22936D0
// Address: 0x22936d0
//
__int64 __fastcall sub_22936D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char *a6,
        int a7,
        __int64 a8,
        __int64 a9)
{
  _QWORD *v11; // r12
  __int64 *v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v23; // edx
  unsigned int v24; // edx
  __int64 v25; // r12
  __int64 v26; // rcx
  unsigned __int64 v27; // rax
  unsigned int v28; // eax
  unsigned __int64 v29; // rax
  unsigned int v30; // eax
  unsigned int *v31; // rdi
  __int64 v32; // rbx
  __int64 v33; // r15
  __int64 v34; // r15
  __int64 v35; // rbx
  unsigned int v36; // eax
  unsigned int v37; // eax
  unsigned int v38; // eax
  __int64 *v39; // rdx
  int v40; // eax
  unsigned int v41; // eax
  char v42; // bl
  __int64 v43; // rdx
  __int64 v44; // rbx
  unsigned __int64 v45; // r12
  __int64 v46; // rbx
  unsigned __int64 v47; // r12
  unsigned __int64 v49; // rax
  unsigned int v50; // eax
  unsigned int v51; // eax
  unsigned int v52; // eax
  unsigned __int64 v53; // rax
  unsigned int v54; // eax
  unsigned int v55; // eax
  unsigned int v56; // eax
  __int64 v57; // rdx
  __int64 v58; // rdx
  unsigned int v59; // eax
  unsigned int v60; // eax
  __int64 v61; // rax
  char v62; // [rsp+8h] [rbp-278h]
  unsigned int v63; // [rsp+50h] [rbp-230h]
  char v64; // [rsp+58h] [rbp-228h]
  unsigned int v65; // [rsp+74h] [rbp-20Ch]
  unsigned __int8 v66; // [rsp+88h] [rbp-1F8h]
  __int64 v67; // [rsp+88h] [rbp-1F8h]
  unsigned __int64 v68; // [rsp+90h] [rbp-1F0h] BYREF
  unsigned int v69; // [rsp+98h] [rbp-1E8h]
  __int64 v70; // [rsp+A0h] [rbp-1E0h] BYREF
  int v71; // [rsp+A8h] [rbp-1D8h]
  unsigned __int64 v72; // [rsp+B0h] [rbp-1D0h] BYREF
  unsigned int v73; // [rsp+B8h] [rbp-1C8h]
  unsigned __int64 v74; // [rsp+C0h] [rbp-1C0h] BYREF
  unsigned int v75; // [rsp+C8h] [rbp-1B8h]
  __int64 v76; // [rsp+D0h] [rbp-1B0h] BYREF
  unsigned int v77; // [rsp+D8h] [rbp-1A8h]
  __int64 v78; // [rsp+E0h] [rbp-1A0h] BYREF
  unsigned int v79; // [rsp+E8h] [rbp-198h]
  unsigned __int64 v80; // [rsp+F0h] [rbp-190h] BYREF
  unsigned int v81; // [rsp+F8h] [rbp-188h]
  unsigned __int64 v82; // [rsp+100h] [rbp-180h] BYREF
  unsigned int v83; // [rsp+108h] [rbp-178h]
  __int64 v84; // [rsp+110h] [rbp-170h] BYREF
  unsigned int v85; // [rsp+118h] [rbp-168h]
  __int64 v86[2]; // [rsp+120h] [rbp-160h] BYREF
  __int64 v87[2]; // [rsp+130h] [rbp-150h] BYREF
  __int64 v88[2]; // [rsp+140h] [rbp-140h] BYREF
  __int64 v89[2]; // [rsp+150h] [rbp-130h] BYREF
  __int64 v90[2]; // [rsp+160h] [rbp-120h] BYREF
  __int64 v91; // [rsp+170h] [rbp-110h] BYREF
  int v92; // [rsp+178h] [rbp-108h]
  __int64 v93; // [rsp+180h] [rbp-100h] BYREF
  int v94; // [rsp+188h] [rbp-F8h]
  __int64 v95; // [rsp+190h] [rbp-F0h] BYREF
  int v96; // [rsp+198h] [rbp-E8h]
  __int64 v97; // [rsp+1A0h] [rbp-E0h] BYREF
  int v98; // [rsp+1A8h] [rbp-D8h]
  unsigned __int64 v99; // [rsp+1B0h] [rbp-D0h] BYREF
  unsigned int v100; // [rsp+1B8h] [rbp-C8h]
  unsigned __int64 v101; // [rsp+1C0h] [rbp-C0h] BYREF
  unsigned int v102; // [rsp+1C8h] [rbp-B8h]
  unsigned __int64 v103; // [rsp+1D0h] [rbp-B0h] BYREF
  unsigned int v104; // [rsp+1D8h] [rbp-A8h]
  unsigned __int64 v105; // [rsp+1E0h] [rbp-A0h] BYREF
  unsigned int v106; // [rsp+1E8h] [rbp-98h]
  _BYTE *v107; // [rsp+1F0h] [rbp-90h] BYREF
  __int64 v108; // [rsp+1F8h] [rbp-88h]
  _BYTE v109[32]; // [rsp+200h] [rbp-80h] BYREF
  _BYTE *v110; // [rsp+220h] [rbp-60h] BYREF
  __int64 v111; // [rsp+228h] [rbp-58h]
  _BYTE v112[80]; // [rsp+230h] [rbp-50h] BYREF

  *(_BYTE *)(a8 + 43) = 0;
  v11 = sub_DCC810(*(__int64 **)(a1 + 8), a5, a4, 0, 0);
  v12 = sub_DCAF50(*(__int64 **)(a1 + 8), a3, 0);
  sub_228CE50(a9, a2, (__int64)v12, (__int64)v11, (__int64)a6);
  v66 = 0;
  if ( *((_WORD *)v11 + 12) || *(_WORD *)(a2 + 24) || *(_WORD *)(a3 + 24) )
    return v66;
  v69 = 1;
  v13 = *(_QWORD *)(a2 + 32);
  v68 = 0;
  v71 = 1;
  v14 = *(_DWORD *)(v13 + 32);
  v70 = 0;
  v73 = 1;
  v72 = 0;
  v75 = v14;
  if ( v14 > 0x40 )
    sub_C43780((__int64)&v74, (const void **)(v13 + 24));
  else
    v74 = *(_QWORD *)(v13 + 24);
  v15 = *(_QWORD *)(a3 + 32);
  v77 = *(_DWORD *)(v15 + 32);
  if ( v77 > 0x40 )
    sub_C43780((__int64)&v76, (const void **)(v15 + 24));
  else
    v76 = *(_QWORD *)(v15 + 24);
  v16 = v11[4];
  v79 = *(_DWORD *)(v16 + 32);
  if ( v79 > 0x40 )
    sub_C43780((__int64)&v78, (const void **)(v16 + 24));
  else
    v78 = *(_QWORD *)(v16 + 24);
  v65 = v75;
  v66 = sub_228B3E0(v75, (__int64 *)&v74, (__int64)&v76, (__int64)&v78, (__int64)&v68, (__int64)&v70, (__int64)&v72);
  if ( !v66 )
  {
    v81 = v65;
    v63 = v65 - 1;
    v64 = (v65 - 1) & 0x3F;
    if ( v65 > 0x40 )
    {
      sub_C43690((__int64)&v80, 1, 1);
      v21 = sub_D95540((__int64)v11);
      v19 = sub_228E3C0(a1, a6, v21);
      if ( !v19 )
      {
        v62 = 0;
        v83 = v65;
        goto LABEL_116;
      }
    }
    else
    {
      v17 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v65) & 1;
      if ( !v65 )
        v17 = 0;
      v80 = v17;
      v18 = sub_D95540((__int64)v11);
      v19 = sub_228E3C0(a1, a6, v18);
      v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v65;
      if ( !v19 )
      {
        v62 = 0;
        v83 = v65;
LABEL_22:
        v24 = v65;
        if ( !v65 )
          v20 = 0;
        v82 = v20;
LABEL_25:
        v25 = 1LL << v64;
        v26 = ~(1LL << v64);
        if ( v24 > 0x40 )
          *(_QWORD *)(v82 + 8LL * (v63 >> 6)) &= v26;
        else
          v82 &= v26;
        v85 = v65;
        if ( v65 > 0x40 )
        {
          sub_C43690((__int64)&v84, 0, 0);
          if ( v85 > 0x40 )
          {
            *(_QWORD *)(v84 + 8LL * (v63 >> 6)) |= v25;
LABEL_30:
            sub_C4A3E0((__int64)v86, (__int64)&v78, (__int64)&v68);
            sub_C472A0((__int64)v87, (__int64)&v70, v86);
            sub_C472A0((__int64)v88, (__int64)&v72, v86);
            v110 = v112;
            v107 = v109;
            v108 = 0x200000000LL;
            v111 = 0x200000000LL;
            sub_C4A3E0((__int64)v89, (__int64)&v76, (__int64)&v68);
            if ( sub_AAD930((__int64)v89, 0) )
            {
              sub_9865C0((__int64)&v101, (__int64)v87);
              if ( v102 > 0x40 )
              {
                sub_C43D10((__int64)&v101);
              }
              else
              {
                v27 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v102) & ~v101;
                if ( !v102 )
                  v27 = 0;
                v101 = v27;
              }
              sub_C46250((__int64)&v101);
              v28 = v102;
              v102 = 0;
              v104 = v28;
              v103 = v101;
              sub_228AEF0((__int64)&v105, (__int64)&v103, (__int64)v89);
              sub_2293630((unsigned int *)&v107, (unsigned __int64)&v105);
              sub_969240((__int64 *)&v105);
              sub_969240((__int64 *)&v103);
              sub_969240((__int64 *)&v101);
              if ( !v62 )
                goto LABEL_36;
              sub_9865C0((__int64)&v101, (__int64)&v80);
              sub_C46B40((__int64)&v101, v87);
              v59 = v102;
              v102 = 0;
              v104 = v59;
              v103 = v101;
              sub_228B180((__int64)&v105, (__int64)&v103, (__int64)v89);
              sub_2293630((unsigned int *)&v110, (unsigned __int64)&v105);
              sub_969240((__int64 *)&v105);
              sub_969240((__int64 *)&v103);
              sub_969240((__int64 *)&v101);
              sub_C4A3E0((__int64)v90, (__int64)&v74, (__int64)&v68);
              if ( !sub_AAD930((__int64)v90, 0) )
                goto LABEL_122;
            }
            else
            {
              sub_9865C0((__int64)&v101, (__int64)v87);
              if ( v102 > 0x40 )
              {
                sub_C43D10((__int64)&v101);
              }
              else
              {
                v49 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v102) & ~v101;
                if ( !v102 )
                  v49 = 0;
                v101 = v49;
              }
              sub_C46250((__int64)&v101);
              v50 = v102;
              v102 = 0;
              v104 = v50;
              v103 = v101;
              sub_228B180((__int64)&v105, (__int64)&v103, (__int64)v89);
              sub_2293630((unsigned int *)&v110, (unsigned __int64)&v105);
              sub_969240((__int64 *)&v105);
              sub_969240((__int64 *)&v103);
              sub_969240((__int64 *)&v101);
              if ( !v62 )
              {
LABEL_36:
                sub_C4A3E0((__int64)v90, (__int64)&v74, (__int64)&v68);
                if ( !sub_AAD930((__int64)v90, 0) )
                {
LABEL_37:
                  sub_9865C0((__int64)&v101, (__int64)v88);
                  if ( v102 > 0x40 )
                  {
                    sub_C43D10((__int64)&v101);
                  }
                  else
                  {
                    v29 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v102) & ~v101;
                    if ( !v102 )
                      v29 = 0;
                    v101 = v29;
                  }
                  sub_C46250((__int64)&v101);
                  v30 = v102;
                  v102 = 0;
                  v104 = v30;
                  v103 = v101;
                  sub_228B180((__int64)&v105, (__int64)&v103, (__int64)v90);
                  v31 = (unsigned int *)&v110;
LABEL_42:
                  sub_2293630(v31, (unsigned __int64)&v105);
                  sub_969240((__int64 *)&v105);
                  sub_969240((__int64 *)&v103);
                  sub_969240((__int64 *)&v101);
                  if ( (_DWORD)v108 )
                  {
                    v32 = (unsigned int)v111;
                    if ( (_DWORD)v111 )
                    {
                      v33 = (__int64)v107;
                      v67 = (__int64)&v107[16 * (unsigned int)v108 - 16];
                      if ( (int)sub_C4C880((__int64)v107, v67) <= 0 )
                        v33 = v67;
                      if ( v85 <= 0x40 && *(_DWORD *)(v33 + 8) <= 0x40u )
                      {
                        v58 = *(_QWORD *)v33;
                        v85 = *(_DWORD *)(v33 + 8);
                        v84 = v58;
                      }
                      else
                      {
                        sub_C43990((__int64)&v84, v33);
                        v32 = (unsigned int)v111;
                      }
                      v34 = (__int64)v110;
                      v35 = (__int64)&v110[16 * v32 - 16];
                      if ( (int)sub_C4C880((__int64)v110, v35) >= 0 )
                        v34 = v35;
                      if ( v83 <= 0x40 && *(_DWORD *)(v34 + 8) <= 0x40u )
                      {
                        v57 = *(_QWORD *)v34;
                        v83 = *(_DWORD *)(v34 + 8);
                        v82 = v57;
                      }
                      else
                      {
                        sub_C43990((__int64)&v82, v34);
                      }
                      v66 = 1;
                      if ( (int)sub_C4C880((__int64)&v84, (__int64)&v82) <= 0 )
                      {
                        v92 = 1;
                        v91 = 0;
                        v94 = 1;
                        v93 = 0;
                        if ( (int)sub_C4C880((__int64)v90, (__int64)v89) <= 0 )
                        {
                          sub_9865C0((__int64)&v99, (__int64)v90);
                          sub_C46B40((__int64)&v99, v89);
                          v55 = v100;
                          v100 = 0;
                          v102 = v55;
                          v101 = v99;
                          sub_C472A0((__int64)&v103, (__int64)&v101, (__int64 *)&v82);
                          sub_9865C0((__int64)&v95, (__int64)v88);
                          sub_C46B40((__int64)&v95, v87);
                          v98 = v96;
                          v96 = 0;
                          v97 = v95;
                          sub_C45EE0((__int64)&v103, &v97);
                          v56 = v104;
                          v104 = 0;
                          v106 = v56;
                          v105 = v103;
                          sub_228AD30((__int64)&v91, (__int64)&v105);
                          sub_969240((__int64 *)&v105);
                          sub_969240(&v97);
                          sub_969240(&v95);
                          sub_969240((__int64 *)&v103);
                          sub_969240((__int64 *)&v101);
                          sub_969240((__int64 *)&v99);
                          sub_9865C0((__int64)&v99, (__int64)v90);
                          sub_C46B40((__int64)&v99, v89);
                          v38 = v100;
                          v39 = &v84;
                          v100 = 0;
                        }
                        else
                        {
                          sub_9865C0((__int64)&v99, (__int64)v90);
                          sub_C46B40((__int64)&v99, v89);
                          v36 = v100;
                          v100 = 0;
                          v102 = v36;
                          v101 = v99;
                          sub_C472A0((__int64)&v103, (__int64)&v101, &v84);
                          sub_9865C0((__int64)&v95, (__int64)v88);
                          sub_C46B40((__int64)&v95, v87);
                          v98 = v96;
                          v96 = 0;
                          v97 = v95;
                          sub_C45EE0((__int64)&v103, &v97);
                          v37 = v104;
                          v104 = 0;
                          v106 = v37;
                          v105 = v103;
                          sub_228AD30((__int64)&v91, (__int64)&v105);
                          sub_969240((__int64 *)&v105);
                          sub_969240(&v97);
                          sub_969240(&v95);
                          sub_969240((__int64 *)&v103);
                          sub_969240((__int64 *)&v101);
                          sub_969240((__int64 *)&v99);
                          sub_9865C0((__int64)&v99, (__int64)v90);
                          sub_C46B40((__int64)&v99, v89);
                          v38 = v100;
                          v100 = 0;
                          v39 = (__int64 *)&v82;
                        }
                        v102 = v38;
                        v101 = v99;
                        sub_C472A0((__int64)&v103, (__int64)&v101, v39);
                        sub_9865C0((__int64)&v95, (__int64)v88);
                        sub_C46B40((__int64)&v95, v87);
                        v40 = v96;
                        v96 = 0;
                        v98 = v40;
                        v97 = v95;
                        sub_C45EE0((__int64)&v103, &v97);
                        v41 = v104;
                        v104 = 0;
                        v106 = v41;
                        v105 = v103;
                        sub_228AD30((__int64)&v93, (__int64)&v105);
                        sub_969240((__int64 *)&v105);
                        sub_969240(&v97);
                        sub_969240(&v95);
                        sub_969240((__int64 *)&v103);
                        sub_969240((__int64 *)&v101);
                        sub_969240((__int64 *)&v99);
                        v106 = v65;
                        if ( v65 > 0x40 )
                          sub_C43690((__int64)&v105, 0, 1);
                        else
                          v105 = 0;
                        v42 = 0;
                        if ( (int)sub_C4C880((__int64)&v91, (__int64)&v105) <= 0 )
                          v42 = 2 * ((int)sub_C4C880((__int64)&v93, (__int64)&v105) >= 0);
                        if ( sub_986F30((__int64)&v91, 0) )
                          v42 |= 4u;
                        if ( sub_AAD930((__int64)&v93, 0) )
                          v42 |= 1u;
                        v43 = 16LL * (unsigned int)(a7 - 1);
                        *(_BYTE *)(v43 + *(_QWORD *)(a8 + 48)) = *(_BYTE *)(v43 + *(_QWORD *)(a8 + 48)) & 7 & v42
                                                               | *(_BYTE *)(v43 + *(_QWORD *)(a8 + 48)) & 0xF8;
                        v66 = (*(_BYTE *)(*(_QWORD *)(a8 + 48) + v43) & 7) == 0;
                        sub_969240((__int64 *)&v105);
                        sub_969240(&v93);
                        sub_969240(&v91);
                      }
                    }
                  }
                  sub_969240(v90);
                  sub_969240(v89);
                  v44 = (__int64)v110;
                  v45 = (unsigned __int64)&v110[16 * (unsigned int)v111];
                  if ( v110 != (_BYTE *)v45 )
                  {
                    do
                    {
                      v45 -= 16LL;
                      if ( *(_DWORD *)(v45 + 8) > 0x40u && *(_QWORD *)v45 )
                        j_j___libc_free_0_0(*(_QWORD *)v45);
                    }
                    while ( v44 != v45 );
                    v45 = (unsigned __int64)v110;
                  }
                  if ( (_BYTE *)v45 != v112 )
                    _libc_free(v45);
                  v46 = (__int64)v107;
                  v47 = (unsigned __int64)&v107[16 * (unsigned int)v108];
                  if ( v107 != (_BYTE *)v47 )
                  {
                    do
                    {
                      v47 -= 16LL;
                      if ( *(_DWORD *)(v47 + 8) > 0x40u && *(_QWORD *)v47 )
                        j_j___libc_free_0_0(*(_QWORD *)v47);
                    }
                    while ( v46 != v47 );
                    v47 = (unsigned __int64)v107;
                  }
                  if ( (_BYTE *)v47 != v109 )
                    _libc_free(v47);
                  sub_969240(v88);
                  sub_969240(v87);
                  sub_969240(v86);
                  sub_969240(&v84);
                  sub_969240((__int64 *)&v82);
                  sub_969240((__int64 *)&v80);
                  goto LABEL_83;
                }
LABEL_106:
                sub_9865C0((__int64)&v101, (__int64)v88);
                if ( v102 > 0x40 )
                {
                  sub_C43D10((__int64)&v101);
                }
                else
                {
                  v53 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v102) & ~v101;
                  if ( !v102 )
                    v53 = 0;
                  v101 = v53;
                }
                sub_C46250((__int64)&v101);
                v54 = v102;
                v102 = 0;
                v104 = v54;
                v103 = v101;
                sub_228AEF0((__int64)&v105, (__int64)&v103, (__int64)v90);
                v31 = (unsigned int *)&v107;
                goto LABEL_42;
              }
              sub_9865C0((__int64)&v101, (__int64)&v80);
              sub_C46B40((__int64)&v101, v87);
              v51 = v102;
              v102 = 0;
              v104 = v51;
              v103 = v101;
              sub_228AEF0((__int64)&v105, (__int64)&v103, (__int64)v89);
              sub_2293630((unsigned int *)&v107, (unsigned __int64)&v105);
              sub_969240((__int64 *)&v105);
              sub_969240((__int64 *)&v103);
              sub_969240((__int64 *)&v101);
              sub_C4A3E0((__int64)v90, (__int64)&v74, (__int64)&v68);
              if ( !sub_AAD930((__int64)v90, 0) )
              {
LABEL_122:
                sub_9865C0((__int64)&v101, (__int64)&v80);
                sub_C46B40((__int64)&v101, v88);
                v60 = v102;
                v102 = 0;
                v104 = v60;
                v103 = v101;
                sub_228AEF0((__int64)&v105, (__int64)&v103, (__int64)v90);
                sub_2293630((unsigned int *)&v107, (unsigned __int64)&v105);
                sub_969240((__int64 *)&v105);
                sub_969240((__int64 *)&v103);
                sub_969240((__int64 *)&v101);
                goto LABEL_37;
              }
            }
            sub_9865C0((__int64)&v101, (__int64)&v80);
            sub_C46B40((__int64)&v101, v88);
            v52 = v102;
            v102 = 0;
            v104 = v52;
            v103 = v101;
            sub_228B180((__int64)&v105, (__int64)&v103, (__int64)v90);
            sub_2293630((unsigned int *)&v110, (unsigned __int64)&v105);
            sub_969240((__int64 *)&v105);
            sub_969240((__int64 *)&v103);
            sub_969240((__int64 *)&v101);
            goto LABEL_106;
          }
        }
        else
        {
          v84 = 0;
        }
        v84 |= v25;
        goto LABEL_30;
      }
    }
    v22 = v19[4];
    if ( v81 <= 0x40 && (v23 = *(_DWORD *)(v22 + 32), v23 <= 0x40) )
    {
      v61 = *(_QWORD *)(v22 + 24);
      v81 = v23;
      v80 = v61;
    }
    else
    {
      sub_C43990((__int64)&v80, v22 + 24);
    }
    v83 = v65;
    if ( v65 <= 0x40 )
    {
      v62 = 1;
      v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v65;
      goto LABEL_22;
    }
    v62 = 1;
LABEL_116:
    sub_C43690((__int64)&v82, -1, 1);
    v24 = v83;
    goto LABEL_25;
  }
LABEL_83:
  sub_969240(&v78);
  sub_969240(&v76);
  if ( v75 > 0x40 && v74 )
    j_j___libc_free_0_0(v74);
  if ( v73 > 0x40 && v72 )
    j_j___libc_free_0_0(v72);
  sub_969240(&v70);
  if ( v69 > 0x40 && v68 )
    j_j___libc_free_0_0(v68);
  return v66;
}
