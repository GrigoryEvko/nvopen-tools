// Function: sub_F36000
// Address: 0xf36000
//
__int64 __fastcall sub_F36000(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        void **a8,
        char a9)
{
  __int64 v11; // rax
  __int64 v12; // r14
  _QWORD *v13; // rbx
  _QWORD *v14; // r15
  _QWORD *v15; // rsi
  void (__fastcall *v16)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 *v20; // r14
  __int64 v21; // r15
  char v22; // di
  char v23; // si
  char i; // dl
  int v25; // ecx
  unsigned int v26; // ecx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rsi
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // rdi
  __int64 *v35; // rdi
  unsigned __int64 v36; // r15
  unsigned __int64 v37; // rax
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rbx
  int v41; // r13d
  unsigned int v42; // r15d
  __int64 *v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // r14
  __int64 *v48; // rax
  __int64 v49; // rsi
  __int64 v50; // rcx
  __int64 v51; // rcx
  unsigned int v52; // eax
  unsigned int v53; // edx
  __int64 v54; // rax
  const void *v55; // r13
  size_t v56; // r15
  __int64 v57; // rcx
  unsigned int v58; // eax
  __int64 v59; // r15
  __int64 *v60; // r14
  __int64 v61; // r12
  __int64 v62; // r13
  _QWORD *v63; // rdi
  char *v64; // rsi
  char *v65; // rax
  __int64 v66; // r8
  __int64 v67; // r9
  int v68; // r10d
  size_t v69; // rdx
  __int64 v70; // rax
  unsigned __int64 v71; // rcx
  __int64 v72; // rdx
  __int64 *v73; // rax
  unsigned __int64 v74; // rax
  char v75; // dl
  __int64 v76; // rdx
  unsigned __int64 v77; // rax
  unsigned __int64 v78; // r9
  char *v79; // rdx
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 *v82; // rdx
  __int64 v83; // rdx
  int v84; // eax
  _QWORD *v85; // [rsp-10h] [rbp-3C0h]
  size_t v86; // [rsp+10h] [rbp-3A0h]
  __int64 v87; // [rsp+10h] [rbp-3A0h]
  __int64 v88; // [rsp+18h] [rbp-398h]
  __int64 v89; // [rsp+20h] [rbp-390h]
  __int64 *v90; // [rsp+20h] [rbp-390h]
  __int64 *v91; // [rsp+28h] [rbp-388h]
  __int64 v92; // [rsp+30h] [rbp-380h]
  __int64 v94[2]; // [rsp+40h] [rbp-370h] BYREF
  __int64 v95; // [rsp+50h] [rbp-360h] BYREF
  __int64 v96; // [rsp+60h] [rbp-350h] BYREF
  __int64 *v97; // [rsp+68h] [rbp-348h]
  __int64 v98; // [rsp+70h] [rbp-340h]
  int v99; // [rsp+78h] [rbp-338h]
  char v100; // [rsp+7Ch] [rbp-334h]
  char v101; // [rsp+80h] [rbp-330h] BYREF
  const char *v102; // [rsp+C0h] [rbp-2F0h] BYREF
  __int64 v103; // [rsp+C8h] [rbp-2E8h]
  _QWORD v104[2]; // [rsp+D0h] [rbp-2E0h] BYREF
  __int16 v105; // [rsp+E0h] [rbp-2D0h]
  __int64 v106; // [rsp+2D0h] [rbp-E0h]
  __int64 v107; // [rsp+2D8h] [rbp-D8h]
  __int64 v108; // [rsp+2E0h] [rbp-D0h]
  __int64 v109; // [rsp+2E8h] [rbp-C8h]
  char v110; // [rsp+2F0h] [rbp-C0h]
  __int64 v111; // [rsp+2F8h] [rbp-B8h]
  char *v112; // [rsp+300h] [rbp-B0h]
  __int64 v113; // [rsp+308h] [rbp-A8h]
  int v114; // [rsp+310h] [rbp-A0h]
  char v115; // [rsp+314h] [rbp-9Ch]
  char v116; // [rsp+318h] [rbp-98h] BYREF
  __int16 v117; // [rsp+358h] [rbp-58h]
  _QWORD *v118; // [rsp+360h] [rbp-50h]
  _QWORD *v119; // [rsp+368h] [rbp-48h]
  __int64 v120; // [rsp+370h] [rbp-40h]

  v92 = a4;
  if ( !a9 )
  {
    v20 = a2;
    v21 = a3;
    v22 = a3;
    v23 = BYTE1(a3);
    for ( i = 0; ; i = 1 )
    {
      if ( !v20 )
        BUG();
      v25 = *((unsigned __int8 *)v20 - 24);
      if ( (_BYTE)v25 != 84 )
      {
        v26 = v25 - 39;
        if ( v26 > 0x38 || ((1LL << v26) & 0x100060000000001LL) == 0 )
          break;
      }
      v20 = (__int64 *)v20[1];
      v23 = 0;
      v22 = 0;
    }
    if ( i )
    {
      LOBYTE(v21) = v22;
      v27 = v21;
      BYTE1(v27) = v23;
      v21 = v27;
    }
    sub_CA0F50(v94, a8);
    if ( v94[1] )
    {
      v102 = (const char *)v94;
      v105 = 260;
    }
    else
    {
      v102 = sub_BD5D20(a1);
      v105 = 773;
      v103 = v83;
      v104[0] = ".split";
    }
    v12 = sub_AA8550((_QWORD *)a1, v20, v21, (__int64)&v102, 0);
    if ( a6 )
    {
      v28 = *(unsigned int *)(a6 + 24);
      v31 = *(_QWORD *)(a6 + 8);
      if ( (_DWORD)v28 )
      {
        v28 = (unsigned int)(v28 - 1);
        v32 = v28 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v33 = (__int64 *)(v31 + 16LL * v32);
        v34 = *v33;
        if ( a1 == *v33 )
        {
LABEL_30:
          v35 = (__int64 *)v33[1];
          if ( v35 )
            sub_D4F330(v35, v12, a6);
        }
        else
        {
          v84 = 1;
          while ( v34 != -4096 )
          {
            v29 = (unsigned int)(v84 + 1);
            v32 = v28 & (v84 + v32);
            v33 = (__int64 *)(v31 + 16LL * v32);
            v34 = *v33;
            if ( a1 == *v33 )
              goto LABEL_30;
            v84 = v29;
          }
        }
      }
    }
    if ( !v92 )
    {
      if ( a5 )
      {
        if ( a1 )
        {
          v51 = (unsigned int)(*(_DWORD *)(a1 + 44) + 1);
          v52 = *(_DWORD *)(a1 + 44) + 1;
        }
        else
        {
          v51 = 0;
          v52 = 0;
        }
        v53 = *(_DWORD *)(a5 + 32);
        if ( v52 < v53 )
        {
          v54 = *(_QWORD *)(*(_QWORD *)(a5 + 24) + 8 * v51);
          if ( v54 )
          {
            v55 = *(const void **)(v54 + 24);
            v56 = 8LL * *(unsigned int *)(v54 + 32);
            v86 = v56;
            if ( v56 )
            {
              v90 = (__int64 *)sub_22077B0(v56);
              v91 = &v90[v56 / 8];
              memcpy(v90, v55, v56);
              v53 = *(_DWORD *)(a5 + 32);
            }
            else
            {
              v91 = 0;
              v90 = 0;
            }
            if ( a1 )
            {
              v57 = (unsigned int)(*(_DWORD *)(a1 + 44) + 1);
              v58 = *(_DWORD *)(a1 + 44) + 1;
            }
            else
            {
              v57 = 0;
              v58 = 0;
            }
            if ( v58 < v53 )
              v92 = *(_QWORD *)(*(_QWORD *)(a5 + 24) + 8 * v57);
            *(_BYTE *)(a5 + 112) = 0;
            v59 = sub_B1B5D0(a5, v12, v92);
            if ( v91 != v90 )
            {
              v88 = v12;
              v60 = v90;
              do
              {
                v61 = *v60;
                *(_BYTE *)(a5 + 112) = 0;
                v62 = *(_QWORD *)(v61 + 8);
                if ( v59 != v62 )
                {
                  v102 = (const char *)v61;
                  v63 = *(_QWORD **)(v62 + 24);
                  v64 = (char *)&v63[*(unsigned int *)(v62 + 32)];
                  v65 = (char *)sub_F33420(v63, (__int64)v64, (__int64 *)&v102);
                  if ( v65 + 8 != v64 )
                  {
                    v69 = v64 - (v65 + 8);
                    v64 = v65 + 8;
                    memmove(v65, v65 + 8, v69);
                    v68 = *(_DWORD *)(v62 + 32);
                  }
                  *(_DWORD *)(v62 + 32) = v68 - 1;
                  *(_QWORD *)(v61 + 8) = v59;
                  v70 = *(unsigned int *)(v59 + 32);
                  v71 = *(unsigned int *)(v59 + 36);
                  if ( v70 + 1 > v71 )
                  {
                    v64 = (char *)(v59 + 40);
                    sub_C8D5F0(v59 + 24, (const void *)(v59 + 40), v70 + 1, 8u, v66, v67);
                    v70 = *(unsigned int *)(v59 + 32);
                  }
                  v72 = *(_QWORD *)(v59 + 24);
                  *(_QWORD *)(v72 + 8 * v70) = v61;
                  ++*(_DWORD *)(v59 + 32);
                  if ( *(_DWORD *)(v61 + 16) != *(_DWORD *)(*(_QWORD *)(v61 + 8) + 16LL) + 1 )
                    sub_F33780(v61, v64, v72, v71, v66, v67);
                }
                ++v60;
              }
              while ( v91 != v60 );
              v12 = v88;
            }
            if ( v90 )
              j_j___libc_free_0(v90, v86);
          }
        }
      }
      goto LABEL_48;
    }
    v102 = (const char *)v104;
    v103 = 0x800000000LL;
    v97 = (__int64 *)&v101;
    v96 = 0;
    v98 = 8;
    v99 = 0;
    v100 = 1;
    sub_F35FA0((__int64)&v102, a1, v12 & 0xFFFFFFFFFFFFFFFBLL, v28, v29, v30);
    v36 = (unsigned int)v103;
    v37 = sub_986580(v12);
    v40 = v37;
    if ( v37 )
    {
      v41 = sub_B46E30(v37);
      v36 += (unsigned int)(2 * v41);
      if ( HIDWORD(v103) >= v36 )
        goto LABEL_35;
    }
    else if ( v36 <= HIDWORD(v103) )
    {
LABEL_44:
      v49 = (__int64)v102;
      sub_FFB3D0(v92, v102, v36);
      if ( !v100 )
        _libc_free(v97, v49);
      if ( v102 != (const char *)v104 )
        _libc_free(v102, v49);
LABEL_48:
      if ( a7 )
      {
        v50 = *(_QWORD *)(v12 + 56);
        if ( v50 )
          v50 -= 24;
        sub_D6DEB0((__int64)a7, a1, v12, v50);
      }
      if ( (__int64 *)v94[0] != &v95 )
        j_j___libc_free_0(v94[0], v95 + 1);
      return v12;
    }
    sub_C8D5F0((__int64)&v102, v104, v36, 0x10u, v38, v39);
    v74 = sub_986580(v12);
    v40 = v74;
    if ( !v74 )
    {
LABEL_98:
      v36 = (unsigned int)v103;
      goto LABEL_44;
    }
    v41 = sub_B46E30(v74);
LABEL_35:
    if ( v41 )
    {
      v89 = v12;
      v42 = 0;
      while ( 1 )
      {
        v47 = sub_B46EC0(v40, v42);
        if ( v100 )
        {
          v48 = v97;
          v43 = &v97[HIDWORD(v98)];
          if ( v97 != v43 )
          {
            while ( v47 != *v48 )
            {
              if ( v43 == ++v48 )
                goto LABEL_93;
            }
            goto LABEL_42;
          }
LABEL_93:
          if ( HIDWORD(v98) < (unsigned int)v98 )
            break;
        }
        sub_C8CC70((__int64)&v96, v47, (__int64)v43, v44, v45, v46);
        if ( v75 )
          goto LABEL_88;
LABEL_42:
        if ( ++v42 == v41 )
        {
          v12 = v89;
          v36 = (unsigned int)v103;
          goto LABEL_44;
        }
      }
      ++HIDWORD(v98);
      *v43 = v47;
      ++v96;
LABEL_88:
      v76 = (unsigned int)v103;
      v77 = v47 & 0xFFFFFFFFFFFFFFFBLL;
      v78 = (unsigned int)v103 + 1LL;
      if ( v78 > HIDWORD(v103) )
      {
        sub_C8D5F0((__int64)&v102, v104, (unsigned int)v103 + 1LL, 0x10u, v45, v78);
        v76 = (unsigned int)v103;
        v77 = v47 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v79 = (char *)&v102[16 * v76];
      *((_QWORD *)v79 + 1) = v77;
      v80 = v77 | 4;
      *(_QWORD *)v79 = v89;
      LODWORD(v103) = v103 + 1;
      v81 = (unsigned int)v103;
      if ( (unsigned __int64)(unsigned int)v103 + 1 > HIDWORD(v103) )
      {
        v87 = v80;
        sub_C8D5F0((__int64)&v102, v104, (unsigned int)v103 + 1LL, 0x10u, v45, (unsigned int)v103 + 1LL);
        v81 = (unsigned int)v103;
        v80 = v87;
      }
      v82 = (__int64 *)&v102[16 * v81];
      v82[1] = v80;
      *v82 = a1;
      LODWORD(v103) = v103 + 1;
      goto LABEL_42;
    }
    goto LABEL_98;
  }
  v108 = a5;
  v102 = (const char *)v104;
  v103 = 0x1000000000LL;
  v106 = 0;
  v107 = 0;
  v109 = 0;
  v110 = 1;
  v111 = 0;
  v112 = &v116;
  v113 = 8;
  v114 = 0;
  v115 = 1;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  if ( a4 )
  {
    v11 = sub_F34FF0(a1, a2, a3, a4, a6, a7, a8);
  }
  else
  {
    v73 = 0;
    if ( a5 )
      v73 = (__int64 *)&v102;
    v11 = sub_F34FF0(a1, a2, a3, (__int64)v73, a6, a7, a8);
  }
  v12 = v11;
  sub_FFCE90(&v102);
  sub_FFD870(&v102);
  sub_FFBC40(&v102);
  v13 = v119;
  v14 = v118;
  v15 = v85;
  if ( v119 != v118 )
  {
    do
    {
      v16 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v14[7];
      *v14 = &unk_49E5048;
      if ( v16 )
      {
        v15 = v14 + 5;
        v16(v14 + 5, v14 + 5, 3);
      }
      *v14 = &unk_49DB368;
      v17 = v14[3];
      if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
        sub_BD60C0(v14 + 1);
      v14 += 9;
    }
    while ( v13 != v14 );
    v14 = v118;
  }
  if ( v14 )
  {
    v15 = (_QWORD *)(v120 - (_QWORD)v14);
    j_j___libc_free_0(v14, v120 - (_QWORD)v14);
  }
  if ( !v115 )
  {
    _libc_free(v112, v15);
    v18 = (__int64)v102;
    if ( v102 == (const char *)v104 )
      return v12;
    goto LABEL_16;
  }
  v18 = (__int64)v102;
  if ( v102 != (const char *)v104 )
LABEL_16:
    _libc_free(v18, v15);
  return v12;
}
