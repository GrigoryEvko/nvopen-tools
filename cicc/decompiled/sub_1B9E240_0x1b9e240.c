// Function: sub_1B9E240
// Address: 0x1b9e240
//
void __fastcall sub_1B9E240(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rax
  int v7; // eax
  __int64 *v8; // rcx
  _DWORD *v9; // rcx
  _DWORD *v10; // rsi
  int v11; // edx
  _DWORD *v12; // rax
  int v13; // r12d
  __int64 v14; // rax
  int v15; // eax
  unsigned int v16; // r12d
  __int64 v17; // r8
  int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rdi
  _BYTE *v22; // r14
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  _QWORD *i; // r12
  int v28; // ebx
  __int64 v29; // rax
  int v30; // r9d
  __int64 v31; // rsi
  int v32; // edx
  unsigned int v33; // ecx
  int *v34; // r10
  int v35; // edi
  int v36; // r8d
  _QWORD *v37; // r9
  __int64 v38; // rax
  __int64 v39; // rdx
  _WORD *v40; // rsi
  __int64 v41; // rbx
  __int64 v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // rbx
  __int64 v45; // rdi
  __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  int v51; // r9d
  int v52; // r8d
  unsigned int v53; // r11d
  int j; // r10d
  int v55; // r10d
  int v56; // r9d
  unsigned int v57; // ebx
  unsigned int v58; // r12d
  __int64 v59; // rdx
  _QWORD *v60; // rax
  __int64 v61; // r13
  __int64 v62; // rdi
  __int64 v63; // rdi
  __int64 v64; // rdx
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  int v68; // r9d
  int v69; // r8d
  int v70; // r9d
  __int64 v71; // rdx
  __int64 k; // rax
  __int64 **v73; // r12
  __int64 v74; // r13
  unsigned int v75; // ebx
  __int64 v76; // rax
  int v77; // r9d
  unsigned int *v78; // r8
  _QWORD *v79; // rax
  unsigned int v80; // edx
  int v81; // edx
  __int64 v82; // [rsp+10h] [rbp-100h]
  __int64 **v83; // [rsp+20h] [rbp-F0h]
  __int64 v84; // [rsp+20h] [rbp-F0h]
  __int64 v85; // [rsp+28h] [rbp-E8h]
  __int64 v86; // [rsp+28h] [rbp-E8h]
  __int64 v87; // [rsp+28h] [rbp-E8h]
  __int64 v88; // [rsp+30h] [rbp-E0h]
  __int64 v89; // [rsp+30h] [rbp-E0h]
  __int64 *v90; // [rsp+30h] [rbp-E0h]
  unsigned int v91; // [rsp+30h] [rbp-E0h]
  __int64 *v92; // [rsp+38h] [rbp-D8h]
  unsigned int v93; // [rsp+38h] [rbp-D8h]
  _QWORD *v94; // [rsp+38h] [rbp-D8h]
  __int64 v95; // [rsp+40h] [rbp-D0h]
  bool v96; // [rsp+48h] [rbp-C8h]
  __int64 v97; // [rsp+48h] [rbp-C8h]
  __int64 *v98; // [rsp+50h] [rbp-C0h]
  unsigned int v99; // [rsp+50h] [rbp-C0h]
  __int64 v100; // [rsp+50h] [rbp-C0h]
  __int64 *v101; // [rsp+50h] [rbp-C0h]
  __int64 *v102; // [rsp+58h] [rbp-B8h]
  __int64 v103; // [rsp+60h] [rbp-B0h]
  unsigned int v104; // [rsp+6Ch] [rbp-A4h]
  __int64 v105[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v106; // [rsp+80h] [rbp-90h]
  _BYTE *v107; // [rsp+90h] [rbp-80h] BYREF
  __int64 v108; // [rsp+98h] [rbp-78h]
  _BYTE v109[16]; // [rsp+A0h] [rbp-70h] BYREF
  unsigned __int64 v110; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v111; // [rsp+B8h] [rbp-58h]
  _WORD v112[40]; // [rsp+C0h] [rbp-50h] BYREF

  v4 = sub_1B8F3E0(*(_QWORD *)(*(_QWORD *)(a1 + 456) + 384LL), a2);
  if ( *(_QWORD *)(v4 + 56) != a2 )
    return;
  v5 = v4;
  v6 = sub_15F2050(a2);
  v103 = sub_1632FA0(v6);
  v98 = (__int64 *)sub_13A4950(a2);
  if ( *(_BYTE *)(a2 + 16) == 54 )
    v92 = *(__int64 **)a2;
  else
    v92 = **(__int64 ***)(a2 - 48);
  v104 = *(_DWORD *)v5;
  v83 = (__int64 **)sub_16463B0(v92, *(_DWORD *)v5 * *(_DWORD *)(a1 + 88));
  v7 = sub_1B8DFF0(a2);
  v95 = sub_1647190(v8, v7);
  v102 = (__int64 *)(a1 + 96);
  sub_1B91520(a1, (__int64 *)(a1 + 96), (__int64)v98);
  v107 = v109;
  v108 = 0x200000000LL;
  do
  {
LABEL_6:
    v9 = *(_DWORD **)(v5 + 24);
    v10 = &v9[4 * *(unsigned int *)(v5 + 40)];
  }
  while ( v9 == v10 );
  while ( 1 )
  {
    v11 = *v9;
    v12 = v9;
    if ( (unsigned int)(*v9 + 0x7FFFFFFF) <= 0xFFFFFFFD )
      break;
    v9 += 4;
  }
  while ( 1 )
  {
    if ( v10 == v12 )
      goto LABEL_6;
    if ( a2 == *((_QWORD *)v12 + 1) )
      break;
    v12 += 4;
    if ( v12 == v10 )
      goto LABEL_6;
    while ( 1 )
    {
      v11 = *v12;
      if ( (unsigned int)(*v12 + 0x7FFFFFFF) <= 0xFFFFFFFD )
        break;
      v12 += 4;
    }
  }
  v13 = v11 - *(_DWORD *)(v5 + 48);
  if ( *(_BYTE *)(v5 + 4) )
    v13 += *(_DWORD *)v5 * (*(_DWORD *)(a1 + 88) - 1);
  v14 = sub_1649C60((__int64)v98);
  v96 = 0;
  if ( *(_BYTE *)(v14 + 16) == 56 )
    v96 = sub_15FA300(v14);
  if ( *(_DWORD *)(a1 + 92) )
  {
    v15 = v13;
    v16 = 0;
    v82 = v5;
    v88 = (unsigned int)-v15;
    do
    {
      v110 = v16;
      v20 = sub_1B9DCB0(a1, v98, (unsigned int *)&v110);
      v21 = *(_QWORD **)(a1 + 120);
      v112[0] = 257;
      v22 = (_BYTE *)v20;
      v23 = sub_1643350(v21);
      v24 = sub_159C470(v23, v88, 0);
      v25 = sub_12815B0(v102, 0, v22, v24, (__int64)&v110);
      v26 = v25;
      if ( v96 )
      {
        v85 = v25;
        sub_15FA2E0(v25, 1);
        v26 = v85;
      }
      v112[0] = 257;
      v17 = sub_12AA3B0(v102, 0x2Fu, v26, v95, (__int64)&v110);
      v19 = (unsigned int)v108;
      if ( (unsigned int)v108 >= HIDWORD(v108) )
      {
        v87 = v17;
        sub_16CD150((__int64)&v107, v109, 0, 8, v17, v18);
        v19 = (unsigned int)v108;
        v17 = v87;
      }
      ++v16;
      *(_QWORD *)&v107[8 * v19] = v17;
      LODWORD(v108) = v108 + 1;
    }
    while ( *(_DWORD *)(a1 + 92) > v16 );
    v5 = v82;
  }
  sub_1B91520(a1, v102, a2);
  v97 = sub_1599EF0(v83);
  if ( *(_BYTE *)(a2 + 16) != 54 )
  {
    v99 = 0;
    for ( i = sub_16463B0(v92, *(_DWORD *)(a1 + 88)); *(_DWORD *)(a1 + 92) > v99; ++v99 )
    {
      v110 = (unsigned __int64)v112;
      v111 = 0x400000000LL;
      if ( v104 )
      {
        v28 = 0;
        do
        {
          v29 = *(unsigned int *)(v5 + 40);
          if ( !(_DWORD)v29 )
LABEL_86:
            BUG();
          v30 = v29 - 1;
          v31 = *(_QWORD *)(v5 + 24);
          v32 = v28 + *(_DWORD *)(v5 + 48);
          v33 = (v29 - 1) & (37 * v32);
          v34 = (int *)(v31 + 16LL * v33);
          v35 = *v34;
          if ( v32 != *v34 )
          {
            v52 = *v34;
            v53 = (v29 - 1) & (37 * v32);
            for ( j = 1; ; ++j )
            {
              if ( v52 == 0x7FFFFFFF )
                goto LABEL_86;
              v53 = v30 & (j + v53);
              v52 = *(_DWORD *)(v31 + 16LL * v53);
              if ( v32 == v52 )
                break;
            }
            v55 = 1;
            while ( v35 != 0x7FFFFFFF )
            {
              v81 = v55 + 1;
              v33 = v30 & (v55 + v33);
              v34 = (int *)(v31 + 16LL * v33);
              v35 = *v34;
              if ( v52 == *v34 )
                goto LABEL_32;
              v55 = v81;
            }
            v34 = (int *)(v31 + 16 * v29);
          }
LABEL_32:
          v37 = (_QWORD *)sub_1B9C240((unsigned int *)a1, *(__int64 **)(*((_QWORD *)v34 + 1) - 48LL), v99);
          if ( *(_BYTE *)(v5 + 4) )
            v37 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 32LL))(a1, v37);
          if ( i != (_QWORD *)*v37 )
            v37 = (_QWORD *)sub_1B970A0(a1, (__int64)v37, (__int64)i, v103);
          v38 = (unsigned int)v111;
          if ( (unsigned int)v111 >= HIDWORD(v111) )
          {
            v94 = v37;
            sub_16CD150((__int64)&v110, v112, 0, 8, v36, (int)v37);
            v38 = (unsigned int)v111;
            v37 = v94;
          }
          ++v28;
          *(_QWORD *)(v110 + 8 * v38) = v37;
          v39 = (unsigned int)(v111 + 1);
          LODWORD(v111) = v111 + 1;
        }
        while ( v104 != v28 );
        v40 = (_WORD *)v110;
      }
      else
      {
        v40 = v112;
        v39 = 0;
      }
      v41 = sub_14C5240(v102, v40, v39);
      v42 = sub_14C4CD0((__int64)v102, *(_DWORD *)(a1 + 88), v104);
      v105[0] = (__int64)"interleaved.vec";
      v106 = 259;
      v86 = sub_14C50F0(v102, v41, v97, v42, (__int64)v105);
      v93 = *(_DWORD *)(v5 + 8);
      v89 = *(_QWORD *)&v107[8 * v99];
      v106 = 257;
      v43 = sub_1648A60(64, 2u);
      v44 = (__int64)v43;
      if ( v43 )
        sub_15F9650((__int64)v43, v86, v89, 0, 0);
      v45 = *(_QWORD *)(a1 + 104);
      if ( v45 )
      {
        v90 = *(__int64 **)(a1 + 112);
        sub_157E9D0(v45 + 40, v44);
        v46 = *v90;
        v47 = *(_QWORD *)(v44 + 24) & 7LL;
        *(_QWORD *)(v44 + 32) = v90;
        v46 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v44 + 24) = v46 | v47;
        *(_QWORD *)(v46 + 8) = v44 + 24;
        *v90 = *v90 & 7 | (v44 + 24);
      }
      sub_164B780(v44, v105);
      sub_12A86E0(v102, v44);
      sub_15F9450(v44, v93);
      sub_1B91780(v5, v44, v48, v49, v50, v51);
      if ( (_WORD *)v110 != v112 )
        _libc_free(v110);
    }
    goto LABEL_47;
  }
  v56 = *(_DWORD *)(a1 + 92);
  v110 = (unsigned __int64)v112;
  v111 = 0x200000000LL;
  if ( v56 )
  {
    v57 = 0;
    do
    {
      v58 = *(_DWORD *)(v5 + 8);
      v59 = *(_QWORD *)&v107[8 * v57];
      v105[0] = (__int64)"wide.vec";
      v100 = v59;
      v106 = 259;
      v60 = sub_1648A60(64, 1u);
      v61 = (__int64)v60;
      if ( v60 )
        sub_15F9210((__int64)v60, *(_QWORD *)(*(_QWORD *)v100 + 24LL), v100, 0, 0, 0);
      v62 = *(_QWORD *)(a1 + 104);
      if ( v62 )
      {
        v101 = *(__int64 **)(a1 + 112);
        sub_157E9D0(v62 + 40, v61);
        v63 = *v101;
        v64 = *(_QWORD *)(v61 + 24) & 7LL;
        *(_QWORD *)(v61 + 32) = v101;
        v63 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v61 + 24) = v63 | v64;
        *(_QWORD *)(v63 + 8) = v61 + 24;
        *v101 = *v101 & 7 | (v61 + 24);
      }
      sub_164B780(v61, v105);
      sub_12A86E0(v102, v61);
      sub_15F8F50(v61, v58);
      sub_1B91780(v5, v61, v65, v66, v67, v68);
      v71 = (unsigned int)v111;
      if ( (unsigned int)v111 >= HIDWORD(v111) )
      {
        sub_16CD150((__int64)&v110, v112, 0, 8, v69, v70);
        v71 = (unsigned int)v111;
      }
      ++v57;
      *(_QWORD *)(v110 + 8 * v71) = v61;
      LODWORD(v111) = v111 + 1;
    }
    while ( *(_DWORD *)(a1 + 92) > v57 );
    if ( !v104 )
      goto LABEL_78;
    goto LABEL_66;
  }
  if ( v104 )
  {
LABEL_66:
    v91 = 0;
    for ( k = sub_1B97720(v5, 0); ; k = sub_1B97720(v5, v91) )
    {
      v73 = (__int64 **)k;
      if ( k )
      {
        v74 = sub_14C4E30((__int64)v102, v91, v104, *(_DWORD *)(a1 + 88));
        if ( *(_DWORD *)(a1 + 92) )
        {
          v75 = 0;
          do
          {
            v105[0] = (__int64)"strided.vec";
            v106 = 259;
            v76 = sub_14C50F0(v102, *(_QWORD *)(v110 + 8LL * v75), v97, v74, (__int64)v105);
            v78 = (unsigned int *)v76;
            if ( v92 != *v73 )
            {
              v84 = v76;
              v79 = sub_16463B0(*v73, *(_DWORD *)(a1 + 88));
              v78 = (unsigned int *)sub_1B970A0(a1, v84, (__int64)v79, v103);
            }
            if ( *(_BYTE *)(v5 + 4) )
              v78 = (unsigned int *)(*(__int64 (__fastcall **)(__int64, unsigned int *))(*(_QWORD *)a1 + 32LL))(a1, v78);
            v80 = v75++;
            sub_1B99BD0((unsigned int *)(a1 + 280), (unsigned __int64)v73, v80, (__int64)v78, v78, v77);
          }
          while ( *(_DWORD *)(a1 + 92) > v75 );
        }
      }
      if ( v104 <= ++v91 )
        break;
    }
LABEL_78:
    if ( (_WORD *)v110 != v112 )
      _libc_free(v110);
  }
LABEL_47:
  if ( v107 != v109 )
    _libc_free((unsigned __int64)v107);
}
