// Function: sub_2923BA0
// Address: 0x2923ba0
//
void __fastcall sub_2923BA0(__int64 **a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 **v7; // rbx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 *v11; // r14
  char v12; // al
  __int64 v13; // rdi
  unsigned int *v14; // r10
  __int64 v15; // r11
  __int64 v16; // r13
  __int64 (__fastcall *v17)(__int64, _BYTE *, __int64, __int64); // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 *v20; // r12
  unsigned int **v21; // rdi
  char v22; // al
  __int64 v23; // rdx
  __int64 v25; // rdx
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // rdx
  char v33; // al
  char *v34; // rdx
  char **v35; // rdx
  char v36; // al
  int v37; // eax
  unsigned int v38; // edx
  __int64 v39; // r12
  int v40; // r14d
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  __int64 *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r8
  __int64 v47; // rdx
  char v48; // al
  char *v49; // rdx
  char **v50; // rdx
  char v51; // al
  __int64 v52; // rsi
  int v53; // eax
  char *v54; // rdx
  __int64 v55; // rax
  unsigned int **v56; // r14
  _QWORD *v57; // rax
  __int64 v58; // r9
  __int64 v59; // r13
  unsigned int *v60; // rax
  __int64 v61; // r14
  unsigned int *v62; // r14
  unsigned int *v63; // rbx
  __int64 v64; // rdx
  unsigned int v65; // esi
  __int64 v66; // rax
  __int64 v67; // r9
  unsigned __int64 v68; // rdx
  unsigned __int8 *v69; // r12
  __int64 *v70; // rax
  __int64 v71; // r11
  unsigned int *v72; // rsi
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // r13
  __int64 v78; // r14
  __int64 v79; // r13
  __int64 v80; // rdx
  unsigned int v81; // esi
  __int64 v82; // rax
  __m128i v83; // rax
  unsigned __int64 v84; // rcx
  unsigned __int64 v85; // rdx
  __int64 v86; // [rsp+0h] [rbp-D0h]
  __int64 v87; // [rsp+0h] [rbp-D0h]
  __int64 v88; // [rsp+0h] [rbp-D0h]
  __int64 v89; // [rsp+8h] [rbp-C8h]
  __int64 v90; // [rsp+10h] [rbp-C0h]
  __int64 v91; // [rsp+18h] [rbp-B8h]
  __int64 v92; // [rsp+20h] [rbp-B0h]
  unsigned int *v93; // [rsp+20h] [rbp-B0h]
  __int64 v94; // [rsp+20h] [rbp-B0h]
  __int64 v95; // [rsp+20h] [rbp-B0h]
  unsigned int *v96; // [rsp+20h] [rbp-B0h]
  unsigned int *v97; // [rsp+20h] [rbp-B0h]
  char v98; // [rsp+28h] [rbp-A8h]
  __int64 v99; // [rsp+28h] [rbp-A8h]
  __int64 v100; // [rsp+28h] [rbp-A8h]
  __int64 v102; // [rsp+30h] [rbp-A0h]
  unsigned int *v103; // [rsp+30h] [rbp-A0h]
  __int64 v104; // [rsp+30h] [rbp-A0h]
  __int64 v105; // [rsp+30h] [rbp-A0h]
  __int64 **v107; // [rsp+38h] [rbp-98h]
  char *v108; // [rsp+40h] [rbp-90h] BYREF
  __int64 v109; // [rsp+48h] [rbp-88h]
  char *v110; // [rsp+50h] [rbp-80h]
  __int16 v111; // [rsp+60h] [rbp-70h]
  __m128i v112; // [rsp+70h] [rbp-60h] BYREF
  char *v113; // [rsp+80h] [rbp-50h]
  __int16 v114; // [rsp+90h] [rbp-40h]

  v7 = a1;
  v8 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned __int8)v8 <= 3u || (_BYTE)v8 == 5 )
    goto LABEL_2;
  if ( (unsigned __int8)v8 > 0x14u )
    goto LABEL_100;
  v23 = 1463376;
  if ( _bittest64(&v23, v8) )
  {
LABEL_2:
    v98 = -1;
    v9 = (unsigned int)sub_AE54E0((__int64)a1[14], (__int64)a1[12], a1[5], *((unsigned int *)a1 + 12))
       | (unsigned __int64)(1LL << *((_BYTE *)a1 + 104));
    v10 = -(__int64)v9 & v9;
    if ( v10 )
    {
      _BitScanReverse64(&v10, v10);
      v98 = 63 - (v10 ^ 0x3F);
    }
    v11 = *a1;
    v12 = *(_BYTE *)(a4 + 32);
    if ( v12 )
    {
      if ( v12 == 1 )
      {
        v108 = ".extract";
        v111 = 259;
      }
      else
      {
        if ( *(_BYTE *)(a4 + 33) == 1 )
        {
          v6 = *(_QWORD *)(a4 + 8);
          v54 = *(char **)a4;
        }
        else
        {
          v54 = (char *)a4;
          v12 = 2;
        }
        v108 = v54;
        v109 = v6;
        v110 = ".extract";
        LOBYTE(v111) = v12;
        HIBYTE(v111) = 3;
      }
    }
    else
    {
      v111 = 256;
    }
    v13 = v11[10];
    v14 = (unsigned int *)v7[1];
    v15 = *((unsigned int *)v7 + 4);
    v16 = *a3;
    v17 = *(__int64 (__fastcall **)(__int64, _BYTE *, __int64, __int64))(*(_QWORD *)v13 + 80LL);
    if ( v17 == sub_92FAE0 )
    {
      if ( *(_BYTE *)v16 > 0x15u )
        goto LABEL_79;
      v18 = *a3;
      v93 = (unsigned int *)v7[1];
      v102 = *((unsigned int *)v7 + 4);
      v19 = sub_AAADB0(v18, v93, v102);
      v15 = v102;
      v14 = v93;
      v20 = (__int64 *)v19;
    }
    else
    {
      v97 = (unsigned int *)v7[1];
      v105 = *((unsigned int *)v7 + 4);
      v82 = v17(v13, (_BYTE *)v16, (__int64)v14, v15);
      v14 = v97;
      v15 = v105;
      v20 = (__int64 *)v82;
    }
    if ( v20 )
    {
LABEL_10:
      v21 = (unsigned int **)*v7;
      v22 = *(_BYTE *)(a4 + 32);
      if ( v22 )
      {
        if ( v22 == 1 )
        {
          v112.m128i_i64[0] = (__int64)".gep";
          v114 = 259;
        }
        else
        {
          if ( *(_BYTE *)(a4 + 33) == 1 )
          {
            v86 = *(_QWORD *)(a4 + 8);
            a4 = *(_QWORD *)a4;
          }
          else
          {
            v22 = 2;
          }
          LOBYTE(v114) = v22;
          v113 = ".gep";
          v112.m128i_i64[0] = a4;
          HIBYTE(v114) = 3;
          v112.m128i_i64[1] = v86;
        }
      }
      else
      {
        v114 = 256;
      }
      v55 = sub_921130(
              v21,
              (__int64)v7[12],
              (__int64)v7[11],
              (_BYTE **)v7[5],
              *((unsigned int *)v7 + 12),
              (__int64)&v112,
              3u);
      v56 = (unsigned int **)*v7;
      v94 = v55;
      v114 = 257;
      v57 = sub_BD2C40(80, unk_3F10A10);
      v59 = (__int64)v57;
      if ( v57 )
        sub_B4D3C0((__int64)v57, (__int64)v20, v94, 0, v98, v58, 0, 0);
      (*(void (__fastcall **)(unsigned int *, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v56[11]
                                                                                                 + 16LL))(
        v56[11],
        v59,
        &v112,
        v56[7],
        v56[8]);
      v60 = *v56;
      v61 = 4LL * *((unsigned int *)v56 + 2);
      if ( v60 != &v60[v61] )
      {
        v107 = v7;
        v62 = &v60[v61];
        v63 = v60;
        do
        {
          v64 = *((_QWORD *)v63 + 1);
          v65 = *v63;
          v63 += 4;
          sub_B99FD0(v59, v65, v64);
        }
        while ( v62 != v63 );
        v7 = v107;
      }
      v66 = v7[11][1];
      if ( (unsigned int)*(unsigned __int8 *)(v66 + 8) - 17 <= 1 )
        v66 = **(_QWORD **)(v66 + 16);
      LODWORD(v109) = sub_AE2980((__int64)v7[14], *(_DWORD *)(v66 + 8) >> 8)[3];
      if ( (unsigned int)v109 > 0x40 )
        sub_C43690((__int64)&v108, 0, 0);
      else
        v108 = 0;
      sub_BB5CE0((__int64)v7[12], v7[5], *((unsigned int *)v7 + 12), (__int64)v7[14], (__int64)&v108, v67, 0, 0);
      if ( v7[15] || v7[16] || v7[17] || v7[18] )
      {
        v68 = (unsigned __int64)v108;
        if ( (unsigned int)v109 > 0x40 )
          v68 = *(_QWORD *)v108;
        sub_E00EB0(&v112, (__int64 *)v7 + 15, v68, v20[1], (__int64)v7[14]);
        sub_B9A100(v59, v112.m128i_i64);
      }
      v69 = sub_BD4CB0(
              (unsigned __int8 *)*(v7[19] - 4),
              (void (__fastcall *)(__int64, unsigned __int8 *))nullsub_96,
              (__int64)&v112);
      if ( *v69 == 60 )
      {
        v83.m128i_i64[0] = sub_9208B0((__int64)v7[14], *(_QWORD *)(*(_QWORD *)(v59 - 64) + 8LL));
        v112 = v83;
        v84 = sub_CA1930(&v112);
        v85 = (unsigned __int64)v108;
        if ( (unsigned int)v109 > 0x40 )
          v85 = *(_QWORD *)v108;
        sub_29228E0((__int64)v69, 1, 8 * v85, v84, (__int64)v7[19], v59, *(_QWORD *)(v59 - 32), *(_QWORD *)(v59 - 64));
      }
      if ( (unsigned int)v109 > 0x40 )
      {
        if ( v108 )
          j_j___libc_free_0_0((unsigned __int64)v108);
      }
      return;
    }
LABEL_79:
    v103 = v14;
    v95 = v15;
    v114 = 257;
    v70 = sub_BD2C40(104, 1u);
    v71 = v95;
    v20 = v70;
    if ( v70 )
    {
      v72 = v103;
      v73 = v95;
      v96 = v103;
      v104 = v71;
      v74 = sub_B501B0(*(_QWORD *)(v16 + 8), v72, v73);
      sub_B44260((__int64)v20, v74, 64, 1u, 0, 0);
      if ( *(v20 - 4) )
      {
        v75 = *(v20 - 3);
        *(_QWORD *)*(v20 - 2) = v75;
        if ( v75 )
          *(_QWORD *)(v75 + 16) = *(v20 - 2);
      }
      *(v20 - 4) = v16;
      v76 = *(_QWORD *)(v16 + 16);
      *(v20 - 3) = v76;
      if ( v76 )
        *(_QWORD *)(v76 + 16) = v20 - 3;
      *(v20 - 2) = v16 + 16;
      *(_QWORD *)(v16 + 16) = v20 - 4;
      v20[9] = (__int64)(v20 + 11);
      v20[10] = 0x400000000LL;
      sub_B50030((__int64)v20, v96, v104, (__int64)&v112);
    }
    (*(void (__fastcall **)(__int64, __int64 *, char **, __int64, __int64))(*(_QWORD *)v11[11] + 16LL))(
      v11[11],
      v20,
      &v108,
      v11[7],
      v11[8]);
    v77 = 16LL * *((unsigned int *)v11 + 2);
    v78 = *v11;
    v79 = v78 + v77;
    while ( v79 != v78 )
    {
      v80 = *(_QWORD *)(v78 + 8);
      v81 = *(_DWORD *)v78;
      v78 += 16;
      sub_B99FD0((__int64)v20, v81, v80);
    }
    goto LABEL_10;
  }
  if ( (_BYTE)v8 == 16 )
  {
    v25 = *(_QWORD *)(a2 + 32);
    if ( (_DWORD)v25 )
    {
      v26 = 0;
      v99 = (unsigned int)v25;
      v27 = *((unsigned int *)a1 + 4);
      do
      {
        if ( v27 + 1 > (unsigned __int64)*((unsigned int *)a1 + 5) )
        {
          sub_C8D5F0((__int64)(a1 + 1), a1 + 3, v27 + 1, 4u, a5, a6);
          v27 = *((unsigned int *)a1 + 4);
        }
        *((_DWORD *)a1[1] + v27) = v26;
        v28 = *a1;
        ++*((_DWORD *)a1 + 4);
        v29 = sub_BCB2D0((_QWORD *)v28[9]);
        v30 = sub_ACD640(v29, v26, 0);
        v32 = *((unsigned int *)a1 + 12);
        if ( v32 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
        {
          v87 = v30;
          sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v32 + 1, 8u, v31, v32 + 1);
          v32 = *((unsigned int *)a1 + 12);
          v30 = v87;
        }
        a1[5][v32] = v30;
        ++*((_DWORD *)a1 + 12);
        v33 = *(_BYTE *)(a4 + 32);
        if ( v33 )
        {
          if ( v33 == 1 )
          {
            v35 = (char **)".";
            v108 = ".";
            v36 = 3;
            v111 = 259;
            v91 = v109;
          }
          else
          {
            if ( *(_BYTE *)(a4 + 33) == 1 )
            {
              v34 = *(char **)a4;
              v92 = *(_QWORD *)(a4 + 8);
            }
            else
            {
              v34 = (char *)a4;
              v33 = 2;
            }
            HIBYTE(v111) = 3;
            v110 = ".";
            v109 = v92;
            v108 = v34;
            v35 = &v108;
            LOBYTE(v111) = v33;
            v36 = 2;
          }
          v112.m128i_i64[0] = (__int64)v35;
          LODWORD(v113) = v26;
          v112.m128i_i64[1] = v91;
          LOBYTE(v114) = v36;
          HIBYTE(v114) = 9;
        }
        else
        {
          v111 = 256;
          v114 = 256;
        }
        ++v26;
        sub_2923BA0(a1, *(_QWORD *)(a2 + 24), a3, &v112);
        v37 = *((_DWORD *)a1 + 4);
        --*((_DWORD *)a1 + 12);
        v27 = (unsigned int)(v37 - 1);
        *((_DWORD *)a1 + 4) = v27;
      }
      while ( v99 != v26 );
    }
    return;
  }
  if ( (_BYTE)v8 != 15 )
LABEL_100:
    BUG();
  v38 = *(_DWORD *)(a2 + 12);
  if ( v38 )
  {
    v39 = 0;
    v40 = 0;
    v41 = *((unsigned int *)a1 + 4);
    v100 = v38;
    v42 = v41 + 1;
    if ( v41 + 1 > (unsigned __int64)*((unsigned int *)a1 + 5) )
      goto LABEL_48;
    while ( 1 )
    {
      *((_DWORD *)a1[1] + v41) = v40;
      v43 = *a1;
      ++*((_DWORD *)a1 + 4);
      v44 = sub_BCB2D0((_QWORD *)v43[9]);
      v45 = sub_ACD640(v44, v39, 0);
      v47 = *((unsigned int *)a1 + 12);
      if ( v47 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
      {
        v88 = v45;
        sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v47 + 1, 8u, v46, v47 + 1);
        v47 = *((unsigned int *)a1 + 12);
        v45 = v88;
      }
      a1[5][v47] = v45;
      ++*((_DWORD *)a1 + 12);
      v48 = *(_BYTE *)(a4 + 32);
      if ( v48 )
      {
        if ( v48 == 1 )
        {
          v50 = (char **)".";
          v108 = ".";
          v111 = 259;
          v51 = 3;
          v90 = v109;
        }
        else
        {
          if ( *(_BYTE *)(a4 + 33) == 1 )
          {
            v49 = *(char **)a4;
            v89 = *(_QWORD *)(a4 + 8);
          }
          else
          {
            v49 = (char *)a4;
            v48 = 2;
          }
          HIBYTE(v111) = 3;
          v110 = ".";
          v109 = v89;
          v108 = v49;
          v50 = &v108;
          LOBYTE(v111) = v48;
          v51 = 2;
        }
        v112.m128i_i64[0] = (__int64)v50;
        LODWORD(v113) = v40;
        v112.m128i_i64[1] = v90;
        LOBYTE(v114) = v51;
        HIBYTE(v114) = 9;
      }
      else
      {
        v111 = 256;
        v114 = 256;
      }
      v52 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8 * v39++);
      sub_2923BA0(a1, v52, a3, &v112);
      v53 = *((_DWORD *)a1 + 4);
      --*((_DWORD *)a1 + 12);
      v41 = (unsigned int)(v53 - 1);
      *((_DWORD *)a1 + 4) = v41;
      if ( v100 == v39 )
        break;
      v42 = v41 + 1;
      v40 = v39;
      if ( v41 + 1 > (unsigned __int64)*((unsigned int *)a1 + 5) )
      {
LABEL_48:
        sub_C8D5F0((__int64)(a1 + 1), a1 + 3, v42, 4u, a5, a6);
        v41 = *((unsigned int *)a1 + 4);
      }
    }
  }
}
