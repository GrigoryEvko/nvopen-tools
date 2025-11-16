// Function: sub_2259720
// Address: 0x2259720
//
__int64 __fastcall sub_2259720(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v3; // ebx
  __int64 v4; // r14
  unsigned int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 *v8; // rdx
  __int64 v9; // rcx
  __int64 *v10; // rdi
  __int64 v11; // r8
  __int64 v12; // rcx
  __int64 *v13; // rcx
  __int64 *v14; // r12
  __int64 v15; // rdx
  _QWORD *v16; // r13
  _QWORD *v17; // r14
  char *v18; // rdi
  __int64 v19; // r15
  unsigned __int8 v20; // al
  int v21; // ebx
  _BYTE **v22; // rdx
  __int64 v23; // rax
  _BYTE *v24; // rsi
  _BYTE *v25; // rdx
  unsigned int v26; // r12d
  __int64 v28; // r9
  int v29; // r13d
  __int64 *v30; // r11
  unsigned int v31; // r12d
  unsigned int v32; // r8d
  __int64 *v33; // rcx
  __int64 v34; // rdi
  int v35; // edx
  __int64 v36; // rdx
  unsigned __int64 v37; // r8
  unsigned __int64 v38; // rdx
  __int64 v39; // rax
  __int64 *v40; // r12
  __int64 *v41; // r13
  unsigned int v42; // eax
  __int64 *v43; // rdi
  __int64 v44; // rcx
  unsigned int v45; // edx
  __int64 *v46; // r10
  __int64 v47; // rdi
  int v48; // eax
  int v49; // r11d
  __int64 *v50; // r8
  _BYTE **v51; // rsi
  unsigned __int8 v52; // al
  __int64 v53; // r15
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rdx
  _QWORD *v58; // r13
  _QWORD *v59; // r14
  char *v60; // rdi
  char *v61; // rdi
  char *v62; // rax
  int v63; // r11d
  int v64; // esi
  __int64 *v65; // rcx
  unsigned int v66; // edx
  __int64 v67; // rdi
  unsigned int v68; // ecx
  __int64 v69; // r8
  int v70; // edi
  __int64 *v71; // rsi
  __int64 *v72; // rdi
  unsigned int v73; // r12d
  int v74; // ecx
  __int64 v75; // rsi
  unsigned __int8 v77; // [rsp+1Dh] [rbp-B3h]
  bool v78; // [rsp+1Eh] [rbp-B2h]
  char v79; // [rsp+1Fh] [rbp-B1h]
  __int64 v80; // [rsp+20h] [rbp-B0h]
  __int64 *v81; // [rsp+28h] [rbp-A8h]
  __int64 v82; // [rsp+28h] [rbp-A8h]
  __int64 v83; // [rsp+28h] [rbp-A8h]
  __int64 v84; // [rsp+28h] [rbp-A8h]
  __int64 v85; // [rsp+28h] [rbp-A8h]
  __int64 v86; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v87; // [rsp+38h] [rbp-98h]
  __int64 v88; // [rsp+40h] [rbp-90h]
  __int64 v89; // [rsp+48h] [rbp-88h]
  __int64 *v90; // [rsp+50h] [rbp-80h] BYREF
  __int64 v91; // [rsp+58h] [rbp-78h]
  _BYTE v92[112]; // [rsp+60h] [rbp-70h] BYREF

  v3 = a3 >> 4;
  v4 = sub_BA8DC0(a2, (__int64)"nvvmir.version", 14);
  v77 = v3 & 1;
  v78 = sub_BA8DC0(a2, (__int64)"llvm.dbg.cu", 11) != 0;
  if ( !v4 )
  {
    v61 = getenv("NVVM_IR_VER_CHK");
    if ( (!v61 || (unsigned int)strtol(v61, 0, 10)) && !(unsigned __int8)sub_22586E0(a1, 1u, 0) )
      return 3;
    if ( v78
      && v77
      && ((v62 = getenv("NVVM_IR_VER_CHK")) == 0 || (unsigned int)strtol(v62, 0, 10))
      && !(unsigned __int8)sub_22588F0(a1, 1u, 0) )
    {
      return 3;
    }
    else
    {
      return 0;
    }
  }
  v5 = 0;
  v90 = (__int64 *)v92;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v91 = 0x800000000LL;
  if ( (unsigned int)sub_B91A00(v4) )
  {
    while ( 1 )
    {
      v6 = sub_B91A10(v4, v5);
      if ( (_DWORD)v88 )
        break;
      v8 = v90;
      v9 = 8LL * (unsigned int)v91;
      v10 = &v90[(unsigned __int64)v9 / 8];
      v11 = v9 >> 3;
      v12 = v9 >> 5;
      if ( !v12 )
        goto LABEL_52;
      v13 = &v90[4 * v12];
      do
      {
        if ( v6 == *v8 )
          goto LABEL_11;
        if ( v6 == v8[1] )
        {
          ++v8;
          goto LABEL_11;
        }
        if ( v6 == v8[2] )
        {
          v8 += 2;
          goto LABEL_11;
        }
        if ( v6 == v8[3] )
        {
          v8 += 3;
          goto LABEL_11;
        }
        v8 += 4;
      }
      while ( v13 != v8 );
      v11 = v10 - v8;
LABEL_52:
      switch ( v11 )
      {
        case 2LL:
          goto LABEL_101;
        case 3LL:
          if ( v6 != *v8 )
          {
            ++v8;
LABEL_101:
            if ( v6 != *v8 )
            {
              ++v8;
LABEL_103:
              if ( v6 != *v8 )
              {
                v38 = (unsigned int)v91 + 1LL;
                if ( v38 > HIDWORD(v91) )
                  goto LABEL_105;
                goto LABEL_56;
              }
            }
          }
LABEL_11:
          if ( v10 == v8 )
            break;
          goto LABEL_12;
        case 1LL:
          goto LABEL_103;
      }
      v38 = (unsigned int)v91 + 1LL;
      if ( v38 <= HIDWORD(v91) )
        goto LABEL_56;
LABEL_105:
      v82 = v6;
      sub_C8D5F0((__int64)&v90, v92, v38, 8u, v11, v7);
      v6 = v82;
      v10 = &v90[(unsigned int)v91];
LABEL_56:
      *v10 = v6;
      v39 = (unsigned int)(v91 + 1);
      LODWORD(v91) = v39;
      if ( (unsigned int)v39 > 8 )
      {
        v40 = v90;
        v41 = &v90[v39];
        while ( (_DWORD)v89 )
        {
          v42 = (v89 - 1) & (((unsigned int)*v40 >> 9) ^ ((unsigned int)*v40 >> 4));
          v43 = (__int64 *)(v87 + 8LL * v42);
          v44 = *v43;
          if ( *v40 != *v43 )
          {
            v63 = 1;
            v46 = 0;
            while ( v44 != -4096 )
            {
              if ( v44 != -8192 || v46 )
                v43 = v46;
              v42 = (v89 - 1) & (v63 + v42);
              v44 = *(_QWORD *)(v87 + 8LL * v42);
              if ( *v40 == v44 )
                goto LABEL_59;
              ++v63;
              v46 = v43;
              v43 = (__int64 *)(v87 + 8LL * v42);
            }
            if ( !v46 )
              v46 = v43;
            ++v86;
            v48 = v88 + 1;
            if ( 4 * ((int)v88 + 1) < (unsigned int)(3 * v89) )
            {
              if ( (int)v89 - HIDWORD(v88) - v48 <= (unsigned int)v89 >> 3 )
              {
                sub_AEE220((__int64)&v86, v89);
                if ( !(_DWORD)v89 )
LABEL_178:
                  JUMPOUT(0x426B71);
                v64 = 1;
                v65 = 0;
                v66 = (v89 - 1) & (((unsigned int)*v40 >> 9) ^ ((unsigned int)*v40 >> 4));
                v46 = (__int64 *)(v87 + 8LL * v66);
                v67 = *v46;
                v48 = v88 + 1;
                if ( *v40 != *v46 )
                {
                  while ( v67 != -4096 )
                  {
                    if ( v67 == -8192 && !v65 )
                      v65 = v46;
                    v66 = (v89 - 1) & (v64 + v66);
                    v46 = (__int64 *)(v87 + 8LL * v66);
                    v67 = *v46;
                    if ( *v40 == *v46 )
                      goto LABEL_123;
                    ++v64;
                  }
                  if ( v65 )
                    v46 = v65;
                }
              }
              goto LABEL_123;
            }
LABEL_62:
            sub_AEE220((__int64)&v86, 2 * v89);
            if ( !(_DWORD)v89 )
              goto LABEL_178;
            v45 = (v89 - 1) & (((unsigned int)*v40 >> 9) ^ ((unsigned int)*v40 >> 4));
            v46 = (__int64 *)(v87 + 8LL * v45);
            v47 = *v46;
            v48 = v88 + 1;
            if ( *v40 != *v46 )
            {
              v49 = 1;
              v50 = 0;
              while ( v47 != -4096 )
              {
                if ( !v50 && v47 == -8192 )
                  v50 = v46;
                v45 = (v89 - 1) & (v49 + v45);
                v46 = (__int64 *)(v87 + 8LL * v45);
                v47 = *v46;
                if ( *v40 == *v46 )
                  goto LABEL_123;
                ++v49;
              }
              if ( v50 )
                v46 = v50;
            }
LABEL_123:
            LODWORD(v88) = v48;
            if ( *v46 != -4096 )
              --HIDWORD(v88);
            *v46 = *v40;
          }
LABEL_59:
          if ( v41 == ++v40 )
            goto LABEL_12;
        }
        ++v86;
        goto LABEL_62;
      }
LABEL_12:
      if ( (unsigned int)sub_B91A00(v4) <= ++v5 )
        goto LABEL_13;
    }
    if ( (_DWORD)v89 )
    {
      v28 = v87;
      v29 = 1;
      v30 = 0;
      v31 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
      v32 = (v89 - 1) & v31;
      v33 = (__int64 *)(v87 + 8LL * v32);
      v34 = *v33;
      if ( v6 == *v33 )
        goto LABEL_12;
      while ( v34 != -4096 )
      {
        if ( v34 != -8192 || v30 )
          v33 = v30;
        v32 = (v89 - 1) & (v29 + v32);
        v34 = *(_QWORD *)(v87 + 8LL * v32);
        if ( v6 == v34 )
          goto LABEL_12;
        ++v29;
        v30 = v33;
        v33 = (__int64 *)(v87 + 8LL * v32);
      }
      if ( !v30 )
        v30 = v33;
      v35 = v88 + 1;
      ++v86;
      if ( 4 * ((int)v88 + 1) < (unsigned int)(3 * v89) )
      {
        if ( (int)v89 - HIDWORD(v88) - v35 <= (unsigned int)v89 >> 3 )
        {
          v85 = v6;
          sub_AEE220((__int64)&v86, v89);
          if ( !(_DWORD)v89 )
          {
LABEL_177:
            LODWORD(v88) = v88 + 1;
            BUG();
          }
          v28 = v87;
          v72 = 0;
          v73 = (v89 - 1) & v31;
          v74 = 1;
          v30 = (__int64 *)(v87 + 8LL * v73);
          v35 = v88 + 1;
          v6 = v85;
          v75 = *v30;
          if ( v85 != *v30 )
          {
            while ( v75 != -4096 )
            {
              if ( !v72 && v75 == -8192 )
                v72 = v30;
              v73 = (v89 - 1) & (v74 + v73);
              v30 = (__int64 *)(v87 + 8LL * v73);
              v75 = *v30;
              if ( v85 == *v30 )
                goto LABEL_46;
              ++v74;
            }
            if ( v72 )
              v30 = v72;
          }
        }
        goto LABEL_46;
      }
    }
    else
    {
      ++v86;
    }
    v83 = v6;
    sub_AEE220((__int64)&v86, 2 * v89);
    if ( !(_DWORD)v89 )
      goto LABEL_177;
    v6 = v83;
    v28 = (unsigned int)(v89 - 1);
    v68 = v28 & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
    v30 = (__int64 *)(v87 + 8LL * v68);
    v35 = v88 + 1;
    v69 = *v30;
    if ( v83 != *v30 )
    {
      v70 = 1;
      v71 = 0;
      while ( v69 != -4096 )
      {
        if ( !v71 && v69 == -8192 )
          v71 = v30;
        v68 = v28 & (v70 + v68);
        v30 = (__int64 *)(v87 + 8LL * v68);
        v69 = *v30;
        if ( v83 == *v30 )
          goto LABEL_46;
        ++v70;
      }
      if ( v71 )
        v30 = v71;
    }
LABEL_46:
    LODWORD(v88) = v35;
    if ( *v30 != -4096 )
      --HIDWORD(v88);
    *v30 = v6;
    v36 = (unsigned int)v91;
    v37 = (unsigned int)v91 + 1LL;
    if ( v37 > HIDWORD(v91) )
    {
      v84 = v6;
      sub_C8D5F0((__int64)&v90, v92, (unsigned int)v91 + 1LL, 8u, v37, v28);
      v36 = (unsigned int)v91;
      v6 = v84;
    }
    v90[v36] = v6;
    LODWORD(v91) = v91 + 1;
    goto LABEL_12;
  }
LABEL_13:
  v14 = v90;
  v79 = 0;
  v81 = &v90[(unsigned int)v91];
  if ( v90 == v81 )
    goto LABEL_92;
  do
  {
    while ( 1 )
    {
      v19 = *v14;
      if ( !*v14 )
      {
LABEL_34:
        v26 = 3;
        v81 = v90;
        goto LABEL_35;
      }
      v20 = *(_BYTE *)(v19 - 16);
      if ( (v20 & 2) != 0 )
      {
        v21 = *(_DWORD *)(v19 - 24);
        if ( ((v21 - 2) & 0xFFFFFFFD) != 0 )
          goto LABEL_34;
        v22 = *(_BYTE ***)(v19 - 32);
        v80 = v19 - 16;
        v23 = 0;
        v24 = *v22;
        if ( **v22 == 1 )
          goto LABEL_31;
      }
      else
      {
        v21 = (*(_WORD *)(v19 - 16) >> 6) & 0xF;
        if ( ((((*(_WORD *)(v19 - 16) >> 6) & 0xF) - 2) & 0xFD) != 0 )
          goto LABEL_34;
        v80 = v19 - 16;
        v51 = (_BYTE **)(v19 - 16 - 8LL * ((v20 >> 2) & 0xF));
        v23 = 0;
        v22 = v51;
        v24 = *v51;
        if ( *v24 == 1 )
        {
LABEL_31:
          v23 = *((_QWORD *)v24 + 17);
          if ( *(_BYTE *)v23 != 17 )
            v23 = 0;
        }
      }
      v25 = v22[1];
      if ( *v25 != 1 )
        goto LABEL_34;
      v15 = *((_QWORD *)v25 + 17);
      if ( *(_BYTE *)v15 != 17 || !v23 )
        goto LABEL_34;
      v16 = *(_QWORD **)(v23 + 24);
      if ( *(_DWORD *)(v23 + 32) > 0x40u )
        v16 = (_QWORD *)*v16;
      v17 = *(_QWORD **)(v15 + 24);
      if ( *(_DWORD *)(v15 + 32) > 0x40u )
        v17 = (_QWORD *)*v17;
      v18 = getenv("NVVM_IR_VER_CHK");
      if ( (!v18 || (unsigned int)strtol(v18, 0, 10))
        && (v16 != (_QWORD *)2 || v17)
        && !(unsigned __int8)sub_22586E0(a1, (unsigned __int64)v16, (unsigned __int64)v17) )
      {
        goto LABEL_34;
      }
      if ( v21 == 4 )
        break;
      if ( v81 == ++v14 )
        goto LABEL_91;
    }
    v52 = *(_BYTE *)(v19 - 16);
    if ( (v52 & 2) != 0 )
    {
      v53 = *(_QWORD *)(v19 - 32);
      v54 = 0;
      v55 = *(_QWORD *)(v53 + 16);
      if ( *(_BYTE *)v55 != 1 )
        goto LABEL_79;
    }
    else
    {
      v53 = v80 - 8LL * ((v52 >> 2) & 0xF);
      v54 = 0;
      v55 = *(_QWORD *)(v53 + 16);
      if ( *(_BYTE *)v55 != 1 )
        goto LABEL_79;
    }
    v54 = *(_QWORD *)(v55 + 136);
    if ( *(_BYTE *)v54 != 17 )
      v54 = 0;
LABEL_79:
    v56 = *(_QWORD *)(v53 + 24);
    if ( *(_BYTE *)v56 != 1 )
      goto LABEL_34;
    v57 = *(_QWORD *)(v56 + 136);
    if ( *(_BYTE *)v57 != 17 || !v54 )
      goto LABEL_34;
    v58 = *(_QWORD **)(v54 + 24);
    if ( *(_DWORD *)(v54 + 32) > 0x40u )
      v58 = (_QWORD *)*v58;
    v59 = *(_QWORD **)(v57 + 24);
    if ( *(_DWORD *)(v57 + 32) > 0x40u )
      v59 = (_QWORD *)*v59;
    v60 = getenv("NVVM_IR_VER_CHK");
    if ( (!v60 || (unsigned int)strtol(v60, 0, 10))
      && (v58 != (_QWORD *)3 || (unsigned __int64)v59 > 2)
      && !(unsigned __int8)sub_22588F0(a1, (unsigned __int64)v58, (unsigned __int64)v59) )
    {
      goto LABEL_34;
    }
    v79 = 1;
    ++v14;
  }
  while ( v81 != v14 );
LABEL_91:
  v81 = v90;
LABEL_92:
  if ( (v77 & v78) == 0 || (v26 = 3, v79) )
    v26 = 0;
LABEL_35:
  if ( v81 != (__int64 *)v92 )
    _libc_free((unsigned __int64)v81);
  sub_C7D6A0(v87, 8LL * (unsigned int)v89, 8);
  return v26;
}
