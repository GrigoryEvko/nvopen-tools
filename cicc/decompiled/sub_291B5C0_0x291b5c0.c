// Function: sub_291B5C0
// Address: 0x291b5c0
//
void __fastcall sub_291B5C0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r15
  unsigned __int64 v10; // rax
  char v11; // r12
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned int **v14; // rdi
  char v15; // al
  __int64 v16; // rax
  unsigned int **v17; // r10
  __int64 v18; // rdx
  char v19; // al
  char *v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // r12
  __int64 v24; // rax
  unsigned int **v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // rdx
  char v30; // al
  char *v31; // rdx
  char **v32; // rdx
  char v33; // al
  int v34; // eax
  unsigned int v35; // edx
  __int64 v36; // r12
  int v37; // r15d
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  unsigned int **v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r8
  __int64 v44; // rdx
  char v45; // al
  char *v46; // rdx
  char **v47; // rdx
  char v48; // al
  __int64 v49; // rsi
  int v50; // eax
  __int64 v51; // rdx
  char v52; // r15
  _QWORD *v53; // rax
  unsigned int **v54; // r10
  __int64 v55; // r12
  unsigned int *v56; // r15
  __int64 v57; // r13
  __int64 v58; // rdx
  unsigned int v59; // esi
  __int64 v60; // rax
  __int64 v61; // r8
  __int64 v62; // r9
  unsigned __int64 v63; // rdx
  __int64 v64; // rax
  unsigned int **v65; // r13
  char v66; // al
  unsigned int *v67; // rdi
  _DWORD *v68; // r10
  __int64 v69; // r11
  __int64 v70; // rbx
  __int64 (__fastcall *v71)(__int64, _BYTE *, _BYTE *, __int64, __int64); // rax
  __int64 v72; // rax
  __int64 v73; // r15
  bool v74; // cc
  _QWORD *v75; // rax
  unsigned int *v76; // rbx
  __int64 v77; // r12
  __int64 v78; // rdx
  unsigned int v79; // esi
  __int64 v80; // rax
  __int64 v81; // [rsp+0h] [rbp-E0h]
  __int64 v82; // [rsp+0h] [rbp-E0h]
  __int64 v83; // [rsp+0h] [rbp-E0h]
  __int64 v84; // [rsp+8h] [rbp-D8h]
  __int64 v85; // [rsp+10h] [rbp-D0h]
  __int64 v86; // [rsp+18h] [rbp-C8h]
  __int64 v87; // [rsp+20h] [rbp-C0h]
  __int64 v88; // [rsp+20h] [rbp-C0h]
  __int64 v89; // [rsp+28h] [rbp-B8h]
  __int64 v90; // [rsp+28h] [rbp-B8h]
  unsigned int **v91; // [rsp+28h] [rbp-B8h]
  unsigned int **v92; // [rsp+28h] [rbp-B8h]
  _DWORD *v93; // [rsp+28h] [rbp-B8h]
  __int64 v94; // [rsp+28h] [rbp-B8h]
  _DWORD *v95; // [rsp+28h] [rbp-B8h]
  __int64 v98; // [rsp+38h] [rbp-A8h]
  const void *v99; // [rsp+38h] [rbp-A8h]
  __int64 v100; // [rsp+38h] [rbp-A8h]
  unsigned __int64 *v101; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v102; // [rsp+48h] [rbp-98h]
  char *v103; // [rsp+50h] [rbp-90h] BYREF
  __int64 v104; // [rsp+58h] [rbp-88h]
  char *v105; // [rsp+60h] [rbp-80h]
  __int16 v106; // [rsp+70h] [rbp-70h]
  __m128i v107; // [rsp+80h] [rbp-60h] BYREF
  char *v108; // [rsp+90h] [rbp-50h]
  __int16 v109; // [rsp+A0h] [rbp-40h]

  v10 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned __int8)v10 <= 3u || (_BYTE)v10 == 5 )
    goto LABEL_2;
  if ( (unsigned __int8)v10 > 0x14u )
    goto LABEL_100;
  v21 = 1463376;
  if ( _bittest64(&v21, v10) )
  {
LABEL_2:
    v11 = -1;
    v12 = (unsigned int)sub_AE54E0(
                          *(_QWORD *)(a1 + 112),
                          *(_QWORD *)(a1 + 96),
                          *(_QWORD **)(a1 + 40),
                          *(unsigned int *)(a1 + 48))
        | (unsigned __int64)(1LL << *(_BYTE *)(a1 + 104));
    v13 = -(__int64)v12 & v12;
    if ( v13 )
    {
      _BitScanReverse64(&v13, v13);
      v11 = 63 - (v13 ^ 0x3F);
    }
    v14 = *(unsigned int ***)a1;
    v15 = *(_BYTE *)(a4 + 32);
    if ( v15 )
    {
      if ( v15 == 1 )
      {
        v107.m128i_i64[0] = (__int64)".gep";
        v109 = 259;
      }
      else
      {
        if ( *(_BYTE *)(a4 + 33) == 1 )
        {
          v6 = *(_QWORD *)(a4 + 8);
          v51 = *(_QWORD *)a4;
        }
        else
        {
          v51 = a4;
          v15 = 2;
        }
        v107.m128i_i64[0] = v51;
        v107.m128i_i64[1] = v6;
        v108 = ".gep";
        LOBYTE(v109) = v15;
        HIBYTE(v109) = 3;
      }
    }
    else
    {
      v109 = 256;
    }
    v16 = sub_921130(
            v14,
            *(_QWORD *)(a1 + 96),
            *(_QWORD *)(a1 + 88),
            *(_BYTE ***)(a1 + 40),
            *(unsigned int *)(a1 + 48),
            (__int64)&v107,
            3u);
    v17 = *(unsigned int ***)a1;
    v18 = v16;
    v19 = *(_BYTE *)(a4 + 32);
    if ( v19 )
    {
      if ( v19 == 1 )
      {
        v103 = ".load";
        v106 = 259;
      }
      else
      {
        if ( *(_BYTE *)(a4 + 33) == 1 )
        {
          v7 = *(_QWORD *)(a4 + 8);
          v20 = *(char **)a4;
        }
        else
        {
          v20 = (char *)a4;
          v19 = 2;
        }
        v103 = v20;
        v104 = v7;
        v105 = ".load";
        LOBYTE(v106) = v19;
        HIBYTE(v106) = 3;
      }
    }
    else
    {
      v106 = 256;
    }
    v91 = v17;
    v88 = v18;
    v109 = 257;
    v52 = v11;
    v53 = sub_BD2C40(80, 1u);
    v54 = v91;
    v55 = (__int64)v53;
    if ( v53 )
    {
      sub_B4D190((__int64)v53, a2, v88, (__int64)&v107, 0, v52, 0, 0);
      v54 = v91;
    }
    v92 = v54;
    (*(void (__fastcall **)(unsigned int *, __int64, char **, unsigned int *, unsigned int *))(*(_QWORD *)v54[11] + 16LL))(
      v54[11],
      v55,
      &v103,
      v54[7],
      v54[8]);
    v56 = *v92;
    v57 = (__int64)&(*v92)[4 * *((unsigned int *)v92 + 2)];
    if ( *v92 != (unsigned int *)v57 )
    {
      do
      {
        v58 = *((_QWORD *)v56 + 1);
        v59 = *v56;
        v56 += 4;
        sub_B99FD0(v55, v59, v58);
      }
      while ( (unsigned int *)v57 != v56 );
    }
    v60 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL);
    if ( (unsigned int)*(unsigned __int8 *)(v60 + 8) - 17 <= 1 )
      v60 = **(_QWORD **)(v60 + 16);
    v102 = sub_AE2980(*(_QWORD *)(a1 + 112), *(_DWORD *)(v60 + 8) >> 8)[3];
    if ( v102 > 0x40 )
      sub_C43690((__int64)&v101, 0, 0);
    else
      v101 = 0;
    if ( (*(_QWORD *)(a1 + 120) || *(_QWORD *)(a1 + 128) || *(_QWORD *)(a1 + 136) || *(_QWORD *)(a1 + 144))
      && (unsigned __int8)sub_BB5CE0(
                            *(_QWORD *)(a1 + 96),
                            *(_QWORD **)(a1 + 40),
                            *(unsigned int *)(a1 + 48),
                            *(_QWORD *)(a1 + 112),
                            (__int64)&v101,
                            v62,
                            0,
                            0) )
    {
      v63 = (unsigned __int64)v101;
      if ( v102 > 0x40 )
        v63 = *v101;
      sub_E00EB0(&v107, (__int64 *)(a1 + 120), v63, *(_QWORD *)(v55 + 8), *(_QWORD *)(a1 + 112));
      sub_B9A100(v55, v107.m128i_i64);
    }
    v64 = *(unsigned int *)(a1 + 160);
    if ( v64 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 164) )
    {
      sub_C8D5F0(a1 + 152, (const void *)(a1 + 168), v64 + 1, 8u, v61, v62);
      v64 = *(unsigned int *)(a1 + 160);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8 * v64) = v55;
    ++*(_DWORD *)(a1 + 160);
    v65 = *(unsigned int ***)a1;
    v66 = *(_BYTE *)(a4 + 32);
    if ( v66 )
    {
      if ( v66 == 1 )
      {
        v103 = ".insert";
        v106 = 259;
      }
      else
      {
        if ( *(_BYTE *)(a4 + 33) == 1 )
        {
          v81 = *(_QWORD *)(a4 + 8);
          a4 = *(_QWORD *)a4;
        }
        else
        {
          v66 = 2;
        }
        LOBYTE(v106) = v66;
        HIBYTE(v106) = 3;
        v103 = (char *)a4;
        v104 = v81;
        v105 = ".insert";
      }
    }
    else
    {
      v106 = 256;
    }
    v67 = v65[10];
    v68 = *(_DWORD **)(a1 + 8);
    v69 = *(unsigned int *)(a1 + 16);
    v70 = *a3;
    v71 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v67 + 88LL);
    if ( v71 == sub_9482E0 )
    {
      if ( *(_BYTE *)v70 > 0x15u || *(_BYTE *)v55 > 0x15u )
        goto LABEL_86;
      v93 = v68;
      v98 = v69;
      v72 = sub_AAAE30(*a3, v55, v68, v69);
      v69 = v98;
      v68 = v93;
      v73 = v72;
    }
    else
    {
      v95 = v68;
      v100 = v69;
      v80 = v71((__int64)v67, (_BYTE *)v70, (_BYTE *)v55, (__int64)v68, v69);
      v68 = v95;
      v69 = v100;
      v73 = v80;
    }
    if ( v73 )
    {
LABEL_76:
      v74 = v102 <= 0x40;
      *a3 = v73;
      if ( !v74 )
      {
        if ( v101 )
          j_j___libc_free_0_0((unsigned __int64)v101);
      }
      return;
    }
LABEL_86:
    v94 = v69;
    v99 = v68;
    v109 = 257;
    v75 = sub_BD2C40(104, unk_3F148BC);
    v73 = (__int64)v75;
    if ( v75 )
    {
      sub_B44260((__int64)v75, *(_QWORD *)(v70 + 8), 65, 2u, 0, 0);
      *(_QWORD *)(v73 + 72) = v73 + 88;
      *(_QWORD *)(v73 + 80) = 0x400000000LL;
      sub_B4FD20(v73, v70, v55, v99, v94, (__int64)&v107);
    }
    (*(void (__fastcall **)(unsigned int *, __int64, char **, unsigned int *, unsigned int *))(*(_QWORD *)v65[11] + 16LL))(
      v65[11],
      v73,
      &v103,
      v65[7],
      v65[8]);
    v76 = *v65;
    v77 = (__int64)&(*v65)[4 * *((unsigned int *)v65 + 2)];
    if ( *v65 != (unsigned int *)v77 )
    {
      do
      {
        v78 = *((_QWORD *)v76 + 1);
        v79 = *v76;
        v76 += 4;
        sub_B99FD0(v73, v79, v78);
      }
      while ( (unsigned int *)v77 != v76 );
    }
    goto LABEL_76;
  }
  if ( (_BYTE)v10 == 16 )
  {
    v22 = *(_QWORD *)(a2 + 32);
    if ( (_DWORD)v22 )
    {
      v23 = 0;
      v89 = (unsigned int)v22;
      v24 = *(unsigned int *)(a1 + 16);
      do
      {
        if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
        {
          sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v24 + 1, 4u, a5, a6);
          v24 = *(unsigned int *)(a1 + 16);
        }
        *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * v24) = v23;
        v25 = *(unsigned int ***)a1;
        ++*(_DWORD *)(a1 + 16);
        v26 = sub_BCB2D0(v25[9]);
        v27 = sub_ACD640(v26, v23, 0);
        v29 = *(unsigned int *)(a1 + 48);
        if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
        {
          v82 = v27;
          sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v29 + 1, 8u, v28, v29 + 1);
          v29 = *(unsigned int *)(a1 + 48);
          v27 = v82;
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v29) = v27;
        ++*(_DWORD *)(a1 + 48);
        v30 = *(_BYTE *)(a4 + 32);
        if ( v30 )
        {
          if ( v30 == 1 )
          {
            v32 = (char **)".";
            v106 = 259;
            v103 = ".";
            v33 = 3;
            v86 = v104;
          }
          else
          {
            if ( *(_BYTE *)(a4 + 33) == 1 )
            {
              v31 = *(char **)a4;
              v87 = *(_QWORD *)(a4 + 8);
            }
            else
            {
              v31 = (char *)a4;
              v30 = 2;
            }
            HIBYTE(v106) = 3;
            v105 = ".";
            v104 = v87;
            v103 = v31;
            v32 = &v103;
            LOBYTE(v106) = v30;
            v33 = 2;
          }
          v107.m128i_i64[0] = (__int64)v32;
          LODWORD(v108) = v23;
          v107.m128i_i64[1] = v86;
          LOBYTE(v109) = v33;
          HIBYTE(v109) = 9;
        }
        else
        {
          v106 = 256;
          v109 = 256;
        }
        ++v23;
        sub_291B5C0(a1, *(_QWORD *)(a2 + 24), a3, &v107);
        v34 = *(_DWORD *)(a1 + 16);
        --*(_DWORD *)(a1 + 48);
        v24 = (unsigned int)(v34 - 1);
        *(_DWORD *)(a1 + 16) = v24;
      }
      while ( v89 != v23 );
    }
    return;
  }
  if ( (_BYTE)v10 != 15 )
LABEL_100:
    BUG();
  v35 = *(_DWORD *)(a2 + 12);
  if ( v35 )
  {
    v36 = 0;
    v37 = 0;
    v38 = *(unsigned int *)(a1 + 16);
    v90 = v35;
    v39 = v38 + 1;
    if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
      goto LABEL_44;
    while ( 1 )
    {
      *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * v38) = v37;
      v40 = *(unsigned int ***)a1;
      ++*(_DWORD *)(a1 + 16);
      v41 = sub_BCB2D0(v40[9]);
      v42 = sub_ACD640(v41, v36, 0);
      v44 = *(unsigned int *)(a1 + 48);
      if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
      {
        v83 = v42;
        sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v44 + 1, 8u, v43, v44 + 1);
        v44 = *(unsigned int *)(a1 + 48);
        v42 = v83;
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v44) = v42;
      ++*(_DWORD *)(a1 + 48);
      v45 = *(_BYTE *)(a4 + 32);
      if ( v45 )
      {
        if ( v45 == 1 )
        {
          v47 = (char **)".";
          v103 = ".";
          v106 = 259;
          v48 = 3;
          v85 = v104;
        }
        else
        {
          if ( *(_BYTE *)(a4 + 33) == 1 )
          {
            v46 = *(char **)a4;
            v84 = *(_QWORD *)(a4 + 8);
          }
          else
          {
            v46 = (char *)a4;
            v45 = 2;
          }
          HIBYTE(v106) = 3;
          v105 = ".";
          v104 = v84;
          v103 = v46;
          v47 = &v103;
          LOBYTE(v106) = v45;
          v48 = 2;
        }
        v107.m128i_i64[0] = (__int64)v47;
        LODWORD(v108) = v37;
        v107.m128i_i64[1] = v85;
        LOBYTE(v109) = v48;
        HIBYTE(v109) = 9;
      }
      else
      {
        v106 = 256;
        v109 = 256;
      }
      v49 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v36++);
      sub_291B5C0(a1, v49, a3, &v107);
      v50 = *(_DWORD *)(a1 + 16);
      --*(_DWORD *)(a1 + 48);
      v38 = (unsigned int)(v50 - 1);
      *(_DWORD *)(a1 + 16) = v38;
      if ( v90 == v36 )
        break;
      v39 = v38 + 1;
      v37 = v36;
      if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
      {
LABEL_44:
        sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v39, 4u, a5, a6);
        v38 = *(unsigned int *)(a1 + 16);
      }
    }
  }
}
