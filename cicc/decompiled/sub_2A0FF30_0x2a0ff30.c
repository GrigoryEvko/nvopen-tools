// Function: sub_2A0FF30
// Address: 0x2a0ff30
//
_BOOL8 __fastcall sub_2A0FF30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        unsigned __int8 a9,
        unsigned int a10,
        char a11,
        char a12)
{
  __int64 v12; // r13
  int v13; // r15d
  __int64 v14; // r12
  char v15; // dl
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _BOOL4 v20; // r15d
  __int64 v22; // rax
  __int64 v23; // r14
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  int v26; // r13d
  unsigned int v27; // r14d
  __int64 v28; // rbx
  __int64 v29; // rsi
  _QWORD *v30; // rax
  _QWORD *v31; // rcx
  __int64 v32; // r14
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r10
  __int64 v37; // r13
  __int64 v38; // rbx
  char v39; // dl
  __int64 v40; // rax
  __int64 v41; // rdi
  _BYTE *v42; // rax
  char **v43; // rsi
  _BYTE *v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  _QWORD *v52; // rbx
  _QWORD *i; // r14
  void (__fastcall *v54)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v55; // rax
  _BYTE **v56; // rax
  __int64 v57; // r13
  __int64 v58; // rsi
  _QWORD *v59; // rax
  _QWORD *v60; // rdx
  __int64 v61; // [rsp-10h] [rbp-3A0h]
  __int64 v62; // [rsp-8h] [rbp-398h]
  __int64 v63; // [rsp+0h] [rbp-390h]
  char v64; // [rsp+Fh] [rbp-381h]
  unsigned __int64 v65; // [rsp+18h] [rbp-378h]
  __int64 v66; // [rsp+18h] [rbp-378h]
  __int64 v67; // [rsp+20h] [rbp-370h]
  __int64 v68; // [rsp+20h] [rbp-370h]
  __int64 v69; // [rsp+28h] [rbp-368h]
  char v70; // [rsp+30h] [rbp-360h]
  __int64 v71; // [rsp+38h] [rbp-358h]
  __int64 v72; // [rsp+38h] [rbp-358h]
  unsigned __int64 v73; // [rsp+38h] [rbp-358h]
  __int64 v74; // [rsp+48h] [rbp-348h] BYREF
  unsigned int v75; // [rsp+50h] [rbp-340h] BYREF
  __int64 v76; // [rsp+58h] [rbp-338h]
  __int64 v77; // [rsp+60h] [rbp-330h]
  __int64 v78; // [rsp+68h] [rbp-328h]
  __int64 v79; // [rsp+70h] [rbp-320h]
  __int64 v80; // [rsp+78h] [rbp-318h]
  __int64 *v81; // [rsp+80h] [rbp-310h]
  __int64 v82; // [rsp+88h] [rbp-308h]
  unsigned __int8 v83; // [rsp+90h] [rbp-300h]
  char v84; // [rsp+91h] [rbp-2FFh]
  char v85; // [rsp+92h] [rbp-2FEh]
  char *v86; // [rsp+A0h] [rbp-2F0h] BYREF
  __int64 v87; // [rsp+A8h] [rbp-2E8h]
  char v88; // [rsp+B0h] [rbp-2E0h] BYREF
  int v89; // [rsp+B8h] [rbp-2D8h]
  __int64 v90; // [rsp+2B0h] [rbp-E0h]
  __int64 v91; // [rsp+2B8h] [rbp-D8h]
  __int64 v92; // [rsp+2C0h] [rbp-D0h]
  __int64 v93; // [rsp+2C8h] [rbp-C8h]
  char v94; // [rsp+2D0h] [rbp-C0h]
  __int64 v95; // [rsp+2D8h] [rbp-B8h]
  char *v96; // [rsp+2E0h] [rbp-B0h]
  __int64 v97; // [rsp+2E8h] [rbp-A8h]
  int v98; // [rsp+2F0h] [rbp-A0h]
  char v99; // [rsp+2F4h] [rbp-9Ch]
  char v100; // [rsp+2F8h] [rbp-98h] BYREF
  __int16 v101; // [rsp+338h] [rbp-58h]
  _QWORD *v102; // [rsp+340h] [rbp-50h]
  _QWORD *v103; // [rsp+348h] [rbp-48h]
  __int64 v104; // [rsp+350h] [rbp-40h]

  v12 = a1;
  v76 = a2;
  v77 = a3;
  v75 = a10;
  v78 = a4;
  v81 = a7;
  v79 = a5;
  v82 = a8;
  v80 = a6;
  v83 = a9;
  v84 = a11;
  v85 = a12;
  v13 = a9;
  v14 = sub_D49300(a1, a2, a3, a4, a5, a6);
  if ( a9 )
  {
    v15 = 0;
    v13 = 0;
    goto LABEL_3;
  }
  v22 = sub_D47930(a1);
  v23 = v22;
  if ( !v22 || (*(_WORD *)(v22 + 2) & 0x7FFF) != 0 )
    goto LABEL_14;
  v24 = *(_QWORD *)(v22 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v24 == v22 + 48 )
    goto LABEL_91;
  if ( !v24 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v24 - 24) - 30 > 0xA )
LABEL_91:
    BUG();
  if ( *(_BYTE *)(v24 - 24) != 31 )
    goto LABEL_14;
  if ( (*(_DWORD *)(v24 - 20) & 0x7FFFFFF) != 1 )
    goto LABEL_14;
  v25 = sub_AA54C0(v22);
  v71 = v25;
  if ( !v25 )
    goto LABEL_14;
  v74 = v25;
  sub_FDC9F0((__int64)&v86, &v74);
  v69 = a1 + 56;
  if ( (_DWORD)v87 == v89 )
    goto LABEL_14;
  v26 = v89;
  v67 = v23;
  v27 = v87;
  v65 = v24;
  v28 = (__int64)v86;
  while ( 1 )
  {
    v29 = sub_B46EC0(v28, v27);
    if ( !*(_BYTE *)(a1 + 84) )
      break;
    v30 = *(_QWORD **)(a1 + 64);
    v31 = &v30[*(unsigned int *)(a1 + 76)];
    if ( v30 == v31 )
      goto LABEL_26;
    while ( v29 != *v30 )
    {
      if ( v31 == ++v30 )
        goto LABEL_26;
    }
LABEL_23:
    if ( v26 == ++v27 )
    {
      v12 = a1;
      v15 = 0;
      goto LABEL_3;
    }
  }
  if ( sub_C8CA60(v69, v29) )
    goto LABEL_23;
LABEL_26:
  v12 = a1;
  v32 = v67;
  v33 = *(_QWORD *)(v71 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v33 == v71 + 48 )
    goto LABEL_88;
  if ( !v33 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v33 - 24) - 30 > 0xA )
LABEL_88:
    BUG();
  if ( *(_BYTE *)(v33 - 24) != 31 )
  {
LABEL_14:
    v15 = 0;
    goto LABEL_3;
  }
  v72 = *(_QWORD *)(v67 + 56);
  v34 = sub_D46F00(a1);
  v35 = v72;
  v68 = v34;
  if ( v65 != v72 )
  {
    v70 = 0;
    v73 = v65;
    v66 = v32;
    v37 = v35;
    do
    {
      v38 = 0;
      if ( v37 )
        v38 = v37 - 24;
      v39 = sub_991A70((unsigned __int8 *)v38, 0, 0, 0, 0, 1u, 0);
      if ( !v39 )
        goto LABEL_42;
      if ( *(_BYTE *)v38 != 85 )
      {
        switch ( *(_BYTE *)v38 )
        {
          case '*':
          case ',':
          case '6':
          case '7':
          case '8':
          case '9':
          case ':':
          case ';':
            goto LABEL_43;
          case '?':
            v56 = (_BYTE **)(v38 + 32 * (1LL - (*(_DWORD *)(v38 + 4) & 0x7FFFFFF)));
            if ( (_BYTE **)v38 == v56 )
              goto LABEL_43;
            break;
          case 'C':
          case 'D':
          case 'E':
            goto LABEL_50;
          default:
            goto LABEL_42;
        }
        while ( **v56 == 17 )
        {
          v56 += 4;
          if ( (_BYTE **)v38 == v56 )
          {
LABEL_43:
            if ( (*(_BYTE *)(v38 + 7) & 0x40) != 0 )
              v41 = *(_QWORD *)(v38 - 8);
            else
              v41 = v38 - 32LL * (*(_DWORD *)(v38 + 4) & 0x7FFFFFF);
            v42 = *(_BYTE **)v41;
            if ( **(_BYTE **)v41 > 0x15u || (v42 = *(_BYTE **)(v41 + 32), *v42 > 0x15u) )
            {
              if ( !v68 && *((_QWORD *)v42 + 2) )
              {
                v64 = v39;
                v63 = v37;
                v57 = *((_QWORD *)v42 + 2);
                do
                {
                  v58 = *(_QWORD *)(*(_QWORD *)(v57 + 24) + 40LL);
                  if ( *(_BYTE *)(a1 + 84) )
                  {
                    v59 = *(_QWORD **)(a1 + 64);
                    v60 = &v59[*(unsigned int *)(a1 + 76)];
                    if ( v59 == v60 )
                      goto LABEL_42;
                    while ( v58 != *v59 )
                    {
                      if ( v60 == ++v59 )
                        goto LABEL_42;
                    }
                  }
                  else if ( !sub_C8CA60(v69, v58) )
                  {
                    goto LABEL_42;
                  }
                  v57 = *(_QWORD *)(v57 + 8);
                }
                while ( v57 );
                v39 = v64;
                v37 = v63;
              }
              if ( !v70 )
              {
                v70 = v39;
                goto LABEL_50;
              }
            }
            break;
          }
        }
LABEL_42:
        v12 = a1;
        v15 = 0;
        goto LABEL_3;
      }
      v40 = *(_QWORD *)(v38 - 32);
      if ( !v40
        || *(_BYTE *)v40
        || *(_QWORD *)(v40 + 24) != *(_QWORD *)(v38 + 80)
        || (*(_BYTE *)(v40 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v40 + 36) - 68) > 3 )
      {
        goto LABEL_42;
      }
LABEL_50:
      v37 = *(_QWORD *)(v37 + 8);
    }
    while ( v73 != v37 );
    v12 = a1;
    v32 = v66;
  }
  v86 = &v88;
  v87 = 0x1000000000LL;
  v101 = 0;
  v43 = &v86;
  v92 = v79;
  v90 = 0;
  v91 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = &v100;
  v97 = 8;
  v98 = 0;
  v99 = 1;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  sub_F39690(v32, (__int64)&v86, v76, v81, 0, 1, 0);
  v46 = v61;
  v47 = v62;
  if ( v80 )
  {
    v43 = 0;
    sub_D9D700(v80, 0);
  }
  if ( v81 )
  {
    v44 = byte_4F8F8E8;
    if ( byte_4F8F8E8[0] )
    {
      v43 = 0;
      nullsub_390();
    }
  }
  sub_FFCE90((__int64)&v86, (__int64)v43, (__int64)v44, v45, v46, v47);
  sub_FFD870((__int64)&v86, (__int64)v43, v48, v49, v50, v51);
  sub_FFBC40((__int64)&v86, (__int64)v43);
  v52 = v103;
  for ( i = v102; v52 != i; i += 9 )
  {
    v54 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))i[7];
    *i = &unk_49E5048;
    if ( v54 )
      v54(i + 5, i + 5, 3);
    *i = &unk_49DB368;
    v55 = i[3];
    if ( v55 != 0 && v55 != -4096 && v55 != -8192 )
      sub_BD60C0(i + 1);
  }
  if ( v102 )
    j_j___libc_free_0((unsigned __int64)v102);
  if ( !v99 )
    _libc_free((unsigned __int64)v96);
  if ( v86 != &v88 )
    _libc_free((unsigned __int64)v86);
  v15 = 1;
  v13 = 1;
LABEL_3:
  v20 = sub_2A0CFD0(&v75, v12, v15) | v13;
  if ( v14 && v20 )
    sub_D49440(v12, v14, v16, v17, v18, v19);
  return v20;
}
