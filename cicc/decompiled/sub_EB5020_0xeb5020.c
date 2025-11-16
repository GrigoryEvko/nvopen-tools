// Function: sub_EB5020
// Address: 0xeb5020
//
__int64 __fastcall sub_EB5020(__int64 a1, __int64 *a2, char a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  int v7; // eax
  __int64 v8; // rdi
  bool v9; // zf
  __int64 v10; // r12
  void (*v11)(void); // rax
  char v12; // al
  __int64 v13; // rdx
  __int64 v14; // r12
  const char *v15; // r13
  __int64 v16; // rdi
  void (*v17)(void); // rax
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 *v20; // rdi
  const char *v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 v25; // rdi
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  __int64 *v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // r15
  __int64 v33; // r12
  __int64 v34; // r10
  int v35; // r13d
  __int64 v36; // rax
  unsigned __int64 v37; // rax
  __int64 *v38; // rdi
  __int64 v39; // r14
  __int64 v40; // r12
  _QWORD *v41; // r15
  void *v42; // rax
  __m128i v43; // xmm0
  __int64 *v44; // rdi
  int v45; // eax
  unsigned int v46; // r12d
  __int64 v48; // rax
  __int64 v49; // r12
  const char *v50; // rax
  __int64 *v51; // rdi
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // r13
  __int64 v55; // r12
  __int64 v56; // rdi
  _QWORD *v57; // r12
  __int64 v58; // rdi
  __int64 v59; // r12
  void (__fastcall *v60)(__int64, _QWORD, __int64); // r13
  __int64 v61; // rax
  __int64 v62; // rax
  int v63; // ecx
  __int64 *v64; // r12
  __int64 *v65; // r13
  const char **v66; // r10
  __int64 v67; // rax
  _QWORD *v68; // rcx
  __int64 v69; // rax
  __int64 *v70; // rdx
  __int64 *v71; // rax
  __int64 v72; // rdx
  __int64 v73; // r13
  __int64 *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rsi
  __int64 *v77; // rcx
  __int64 v78; // rax
  unsigned __int64 v79; // rax
  __int64 *v80; // rdi
  char v81; // [rsp+3Bh] [rbp-325h]
  int v83; // [rsp+58h] [rbp-308h]
  char v84; // [rsp+5Fh] [rbp-301h]
  __int64 v85; // [rsp+78h] [rbp-2E8h]
  __int64 v86; // [rsp+78h] [rbp-2E8h]
  __int64 v87; // [rsp+78h] [rbp-2E8h]
  const char **v88; // [rsp+78h] [rbp-2E8h]
  __int64 v89; // [rsp+80h] [rbp-2E0h] BYREF
  __int64 v90; // [rsp+88h] [rbp-2D8h]
  const char *v91; // [rsp+90h] [rbp-2D0h] BYREF
  __int64 v92; // [rsp+98h] [rbp-2C8h]
  _QWORD *v93; // [rsp+A0h] [rbp-2C0h]
  __int64 v94; // [rsp+A8h] [rbp-2B8h]
  __int16 v95; // [rsp+B0h] [rbp-2B0h]
  const char *v96; // [rsp+C0h] [rbp-2A0h] BYREF
  __int64 v97; // [rsp+C8h] [rbp-298h]
  _QWORD v98[2]; // [rsp+D0h] [rbp-290h] BYREF
  __int16 v99; // [rsp+E0h] [rbp-280h]
  int v100; // [rsp+110h] [rbp-250h]
  char v101; // [rsp+114h] [rbp-24Ch]
  _QWORD *v102; // [rsp+118h] [rbp-248h]
  _QWORD v103[2]; // [rsp+120h] [rbp-240h] BYREF
  _BYTE v104[560]; // [rsp+130h] [rbp-230h] BYREF

  *(_DWORD *)(a1 + 776) = 0;
  sub_EA2BF0(*(_QWORD *)(a1 + 832));
  *(_QWORD *)(a1 + 832) = 0;
  *(_QWORD *)(a1 + 840) = a1 + 824;
  *(_QWORD *)(a1 + 848) = a1 + 824;
  *(_QWORD *)(a1 + 856) = 0;
  if ( !(_BYTE)a2 )
  {
    v59 = *(_QWORD *)(a1 + 232);
    v60 = *(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v59 + 192LL);
    v61 = sub_ECE6C0(*(_QWORD *)(a1 + 8));
    a2 = 0;
    v60(v59, 0, v61);
  }
  sub_EABFE0(a1);
  v7 = *(_DWORD *)(a1 + 308);
  v8 = *(_QWORD *)(a1 + 224);
  *(_BYTE *)(a1 + 32) = 0;
  v83 = v7;
  v9 = *(_BYTE *)(v8 + 1793) == 0;
  v81 = *(_BYTE *)(a1 + 313);
  v103[0] = v104;
  v103[1] = 0x400000000LL;
  if ( !v9 )
  {
    v10 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 232) + 288LL) + 8LL);
    if ( !*(_QWORD *)(v10 + 16) )
    {
      v73 = sub_E6C430(v8, (__int64)a2, v4, v5, v6);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 232) + 208LL))(*(_QWORD *)(a1 + 232), v73, 0);
      *(_QWORD *)(v10 + 16) = v73;
      v8 = *(_QWORD *)(a1 + 224);
    }
    a2 = (__int64 *)&v96;
    v96 = (const char *)v10;
    sub_EAB3D0(v8 + 1800, (__int64 *)&v96);
  }
  v11 = *(void (**)(void))(**(_QWORD **)(a1 + 8) + 200LL);
  if ( v11 != nullsub_372 )
    v11();
  if ( **(_DWORD **)(a1 + 48) )
  {
    while ( 1 )
    {
      v100 = -1;
      v101 = 0;
      a2 = (__int64 *)&v96;
      v96 = (const char *)v98;
      v97 = 0x800000000LL;
      v102 = v103;
      v12 = (*(__int64 (__fastcall **)(__int64, const char **, _QWORD))(*(_QWORD *)a1 + 272LL))(a1, &v96, 0);
      v13 = *(unsigned int *)(a1 + 24);
      v84 = v12;
      if ( !v12 || (_DWORD)v13 )
        goto LABEL_58;
      if ( **(_DWORD **)(a1 + 48) == 1 )
        break;
      if ( !*(_BYTE *)(a1 + 155) )
        goto LABEL_67;
LABEL_14:
      v14 = (__int64)v96;
      v15 = &v96[8 * (unsigned int)v97];
      if ( v96 != v15 )
      {
        do
        {
          v16 = *((_QWORD *)v15 - 1);
          v15 -= 8;
          if ( v16 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
        }
        while ( (const char *)v14 != v15 );
        v15 = v96;
      }
      if ( v15 != (const char *)v98 )
        _libc_free(v15, a2);
      if ( !**(_DWORD **)(a1 + 48) )
        goto LABEL_22;
    }
    sub_EABFE0(a1);
    v13 = *(unsigned int *)(a1 + 24);
LABEL_58:
    v48 = *(_QWORD *)(a1 + 16);
    v87 = v48 + 112 * v13;
    if ( v48 != v87 )
    {
      v49 = *(_QWORD *)(a1 + 16);
      do
      {
        v95 = 261;
        v50 = *(const char **)(v49 + 8);
        v49 += 112;
        v51 = *(__int64 **)(a1 + 248);
        v91 = v50;
        v92 = *(_QWORD *)(v49 - 96);
        a2 = *(__int64 **)(v49 - 112);
        v52 = *(_QWORD *)(v49 - 16);
        v53 = *(_QWORD *)(v49 - 8);
        *(_BYTE *)(a1 + 32) = 1;
        v89 = v52;
        v90 = v53;
        sub_C91CB0(v51, (unsigned __int64)a2, 0, (__int64)&v91, (__int64)&v89, 1, 0, 0, 1u);
        sub_EA2AE0((_QWORD *)a1);
      }
      while ( v87 != v49 );
      v54 = *(_QWORD *)(a1 + 16);
      v55 = v54 + 112LL * *(unsigned int *)(a1 + 24);
      while ( v54 != v55 )
      {
        while ( 1 )
        {
          v55 -= 112;
          v56 = *(_QWORD *)(v55 + 8);
          if ( v56 == v55 + 32 )
            break;
          _libc_free(v56, a2);
          if ( v54 == v55 )
            goto LABEL_65;
        }
      }
    }
LABEL_65:
    *(_DWORD *)(a1 + 24) = 0;
    if ( !v84 || *(_BYTE *)(a1 + 155) )
      goto LABEL_14;
LABEL_67:
    sub_EB4E00(a1);
    goto LABEL_14;
  }
LABEL_22:
  v17 = *(void (**)(void))(**(_QWORD **)(a1 + 8) + 208LL);
  if ( v17 != nullsub_373 )
    v17();
  v18 = *(_QWORD *)(a1 + 16);
  if ( v18 != v18 + 112LL * *(unsigned int *)(a1 + 24) )
  {
    v85 = v18 + 112LL * *(unsigned int *)(a1 + 24);
    do
    {
      v99 = 261;
      v19 = *(_QWORD *)(v18 + 8);
      v18 += 112;
      v20 = *(__int64 **)(a1 + 248);
      v96 = (const char *)v19;
      v97 = *(_QWORD *)(v18 - 96);
      v21 = *(const char **)(v18 - 16);
      a2 = *(__int64 **)(v18 - 112);
      v22 = *(_QWORD *)(v18 - 8);
      *(_BYTE *)(a1 + 32) = 1;
      v91 = v21;
      v92 = v22;
      sub_C91CB0(v20, (unsigned __int64)a2, 0, (__int64)&v96, (__int64)&v91, 1, 0, 0, 1u);
      sub_EA2AE0((_QWORD *)a1);
    }
    while ( v85 != v18 );
    v23 = *(_QWORD *)(a1 + 16);
    v24 = v23 + 112LL * *(unsigned int *)(a1 + 24);
    while ( v23 != v24 )
    {
      while ( 1 )
      {
        v24 -= 112;
        v25 = *(_QWORD *)(v24 + 8);
        if ( v25 == v24 + 32 )
          break;
        _libc_free(v25, a2);
        if ( v23 == v24 )
          goto LABEL_31;
      }
    }
  }
LABEL_31:
  *(_DWORD *)(a1 + 24) = 0;
  if ( v83 != *(_DWORD *)(a1 + 308) || *(_BYTE *)(a1 + 313) != v81 )
  {
    v96 = "unmatched .ifs or .elses";
    v99 = 259;
    v26 = sub_ECD7B0(a1);
    v27 = sub_ECD6A0(v26);
    *(_BYTE *)(a1 + 32) = 1;
    v28 = *(__int64 **)(a1 + 248);
    a2 = (__int64 *)v27;
    v91 = 0;
    v92 = 0;
    sub_C91CB0(v28, v27, 0, (__int64)&v96, (__int64)&v91, 1, 0, 0, 1u);
    sub_EA2AE0((_QWORD *)a1);
  }
  v29 = *(_QWORD *)(a1 + 224);
  if ( *(_QWORD *)(v29 + 1768) )
  {
    v30 = *(_QWORD *)(v29 + 1752);
    v31 = *(_QWORD *)(v30 + 160);
    v32 = v31 + 80LL * *(unsigned int *)(v30 + 168);
    if ( v31 != v32 )
    {
      v33 = v31 + 80;
      v34 = v32;
      v35 = 0;
      while ( 1 )
      {
        ++v35;
        if ( v34 == v33 )
          break;
        if ( !*(_QWORD *)(v33 + 8) && v35 )
        {
          v91 = "unassigned file number: ";
          v86 = v34;
          v96 = (const char *)&v91;
          v95 = 2307;
          v99 = 770;
          LODWORD(v93) = v35;
          v98[0] = " for .file directives";
          v36 = sub_ECD7B0(a1);
          v37 = sub_ECD6A0(v36);
          *(_BYTE *)(a1 + 32) = 1;
          v38 = *(__int64 **)(a1 + 248);
          a2 = (__int64 *)v37;
          v89 = 0;
          v90 = 0;
          sub_C91CB0(v38, v37, 0, (__int64)&v96, (__int64)&v89, 1, 0, 0, 1u);
          sub_EA2AE0((_QWORD *)a1);
          v34 = v86;
        }
        v33 += 80;
      }
    }
  }
  if ( !a3 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 18LL) )
    {
      v62 = *(_QWORD *)(a1 + 224);
      v63 = *(_DWORD *)(v62 + 1352);
      if ( v63 )
      {
        a2 = *(__int64 **)(v62 + 1344);
        if ( *a2 == -8 || !*a2 )
        {
          v74 = a2 + 1;
          do
          {
            do
            {
              v75 = *v74;
              v64 = v74++;
            }
            while ( !v75 );
          }
          while ( v75 == -8 );
        }
        else
        {
          v64 = *(__int64 **)(v62 + 1344);
        }
        v65 = &a2[v63];
        if ( v64 != v65 )
        {
          v66 = &v91;
          do
          {
            while ( 1 )
            {
              v67 = *(_QWORD *)(*v64 + 8);
              if ( v67 )
              {
                if ( (*(_BYTE *)(v67 + 8) & 2) != 0 && (*(_BYTE *)(v67 + 9) & 0x70) != 0x20 )
                {
                  v68 = *(_QWORD **)v67;
                  if ( !*(_QWORD *)v67 )
                  {
                    v76 = 0;
                    if ( (*(_BYTE *)(v67 + 8) & 1) != 0 )
                    {
                      v77 = *(__int64 **)(v67 - 8);
                      v76 = *v77;
                      v68 = v77 + 3;
                    }
                    v96 = (const char *)v66;
                    v88 = v66;
                    v98[0] = "' not defined";
                    v93 = v68;
                    v94 = v76;
                    v99 = 770;
                    v95 = 1283;
                    v91 = "assembler local symbol '";
                    v78 = sub_ECD7B0(a1);
                    v79 = sub_ECD6A0(v78);
                    *(_BYTE *)(a1 + 32) = 1;
                    v80 = *(__int64 **)(a1 + 248);
                    a2 = (__int64 *)v79;
                    v89 = 0;
                    v90 = 0;
                    sub_C91CB0(v80, v79, 0, (__int64)&v96, (__int64)&v89, 1, 0, 0, 1u);
                    sub_EA2AE0((_QWORD *)a1);
                    v66 = v88;
                  }
                }
              }
              v69 = v64[1];
              v70 = v64 + 1;
              if ( v69 == -8 || !v69 )
                break;
              ++v64;
              if ( v70 == v65 )
                goto LABEL_44;
            }
            v71 = v64 + 2;
            do
            {
              do
              {
                v72 = *v71;
                v64 = v71++;
              }
              while ( !v72 );
            }
            while ( v72 == -8 );
          }
          while ( v64 != v65 );
        }
      }
    }
LABEL_44:
    v39 = *(_QWORD *)(a1 + 528);
    v40 = v39 + 56LL * *(unsigned int *)(a1 + 536);
    while ( v40 != v39 )
    {
      while ( 1 )
      {
        v41 = *(_QWORD **)v39;
        if ( !**(_QWORD **)v39 )
        {
          if ( (*((_BYTE *)v41 + 9) & 0x70) != 0x20 )
            break;
          if ( *((char *)v41 + 8) < 0 )
            break;
          *((_BYTE *)v41 + 8) |= 8u;
          v42 = sub_E807D0(v41[3]);
          *v41 = v42;
          if ( !v42 )
            break;
        }
        v39 += 56;
        if ( v40 == v39 )
          goto LABEL_52;
      }
      v43 = _mm_loadu_si128((const __m128i *)(v39 + 8));
      v39 += 56;
      v44 = *(__int64 **)(a1 + 248);
      *(__m128i *)(a1 + 480) = v43;
      *(__m128i *)(a1 + 496) = _mm_loadu_si128((const __m128i *)(v39 - 32));
      v45 = *(_DWORD *)(v39 - 16);
      HIBYTE(v99) = 1;
      *(_DWORD *)(a1 + 512) = v45;
      v96 = "directional label undefined";
      LOBYTE(v99) = 3;
      a2 = *(__int64 **)(v39 - 8);
      *(_BYTE *)(a1 + 32) = 1;
      v91 = 0;
      v92 = 0;
      sub_C91CB0(v44, (unsigned __int64)a2, 0, (__int64)&v96, (__int64)&v91, 1, 0, 0, 1u);
      sub_EA2AE0((_QWORD *)a1);
    }
LABEL_52:
    if ( *(_BYTE *)(a1 + 32) )
      goto LABEL_53;
    v57 = *(_QWORD **)(a1 + 232);
    v58 = v57[2];
    if ( v58 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v58 + 72LL))(v58);
      v57 = *(_QWORD **)(a1 + 232);
    }
    a2 = (__int64 *)sub_ECD690(a1 + 40);
    sub_E99F70(v57, a2);
  }
  if ( *(_BYTE *)(a1 + 32) )
  {
LABEL_53:
    v46 = 1;
    goto LABEL_54;
  }
  v46 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 224) + 2376LL);
LABEL_54:
  if ( (_BYTE *)v103[0] != v104 )
    _libc_free(v103[0], a2);
  return v46;
}
