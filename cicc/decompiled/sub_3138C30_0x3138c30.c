// Function: sub_3138C30
// Address: 0x3138c30
//
__int64 __fastcall sub_3138C30(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        const char *a4,
        unsigned __int64 a5,
        const char *a6,
        __int64 a7,
        __int64 a8,
        const char *a9,
        __int64 a10)
{
  __int64 *v12; // r15
  __int64 v13; // rcx
  __int64 v14; // rbx
  unsigned __int8 *v15; // rdi
  unsigned int v16; // r12d
  __int16 v17; // dx
  __int64 v18; // r8
  char v19; // al
  char v20; // dl
  __int16 v21; // cx
  __int64 *v22; // r14
  __int64 v23; // rbx
  unsigned __int8 v24; // al
  unsigned int v25; // ebx
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // r12
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // r14
  __int64 v32; // r15
  _QWORD *v33; // rax
  __int64 v34; // rbx
  __int64 v35; // rax
  char v36; // al
  __int16 v37; // cx
  _QWORD *v38; // rax
  __int64 v39; // r9
  __int64 v40; // r12
  __int64 v41; // r15
  __int64 v42; // rbx
  __int64 v43; // rdx
  unsigned int v44; // esi
  __int64 **v45; // r12
  __int64 v46; // r14
  unsigned int v47; // ebx
  unsigned int v48; // eax
  __int64 v49; // rdi
  __int64 (__fastcall *v50)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v51; // rax
  _QWORD *v52; // rdi
  __int64 v53; // rax
  __int64 **v54; // rcx
  unsigned __int64 v55; // rax
  _QWORD *v56; // rdi
  __int64 v57; // rax
  unsigned __int64 v58; // rsi
  _BYTE *v59; // rdx
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // rdi
  __int64 v63; // r12
  __int64 v64; // rbx
  __int64 v65; // rax
  _QWORD *v66; // rax
  __int64 v67; // r14
  __int64 v68; // r12
  __int64 v69; // rbx
  __int64 v70; // rdx
  unsigned int v71; // esi
  __int64 v72; // rax
  char v73; // r12
  _QWORD *v74; // rax
  __int64 v75; // r9
  __int64 v76; // rbx
  __int64 v77; // r12
  __int64 v78; // r13
  __int64 v79; // rdx
  unsigned int v80; // esi
  _QWORD **v81; // rbx
  __int64 result; // rax
  __int64 v83; // r12
  _QWORD *v84; // rdi
  __int64 v85; // rax
  __int64 v86; // rdi
  __int64 (__fastcall *v87)(__int64, __int64, __int64); // rax
  __int64 v88; // rax
  __int64 v89; // rbx
  __int64 v90; // r14
  __int64 v91; // r12
  __int64 v92; // rdx
  unsigned int v93; // esi
  __int64 v94; // rax
  __int64 v95; // rbx
  __int64 v96; // r14
  __int64 v97; // r12
  __int64 v98; // rdx
  unsigned int v99; // esi
  __int64 v100; // rax
  __int64 v101; // r12
  __int64 v102; // rbx
  __int64 v103; // rdx
  unsigned int v104; // esi
  __int64 v106; // [rsp+10h] [rbp-150h]
  __int64 v110; // [rsp+38h] [rbp-128h]
  __int64 v111; // [rsp+38h] [rbp-128h]
  unsigned __int8 v112; // [rsp+40h] [rbp-120h]
  unsigned int v113; // [rsp+40h] [rbp-120h]
  unsigned int v114; // [rsp+40h] [rbp-120h]
  __int64 v115; // [rsp+48h] [rbp-118h]
  __int64 v116; // [rsp+50h] [rbp-110h]
  __int64 *v117; // [rsp+58h] [rbp-108h]
  __int64 v118; // [rsp+60h] [rbp-100h]
  __int64 v119; // [rsp+68h] [rbp-F8h]
  __int64 *v120; // [rsp+68h] [rbp-F8h]
  char v121; // [rsp+70h] [rbp-F0h]
  __int16 v122; // [rsp+7Ah] [rbp-E6h]
  __int16 v123; // [rsp+7Ch] [rbp-E4h]
  int v124[8]; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v125; // [rsp+A0h] [rbp-C0h]
  _BYTE v126[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int16 v127; // [rsp+D0h] [rbp-90h]
  const char *v128[4]; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v129; // [rsp+100h] [rbp-60h]
  unsigned __int64 v130; // [rsp+108h] [rbp-58h]
  __int64 v131; // [rsp+110h] [rbp-50h]
  __int64 v132; // [rsp+118h] [rbp-48h]
  __int64 v133; // [rsp+120h] [rbp-40h]

  v12 = (__int64 *)(a1 + 512);
  sub_B2D3C0(a2, 0, 22);
  sub_B2D3C0(a2, 1, 22);
  sub_B2D3C0(a2, 0, 40);
  sub_B2D3C0(a2, 1, 40);
  sub_B2CD30(a2, 41);
  v13 = *(_QWORD *)(a2 + 104);
  v14 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL);
  v15 = *(unsigned __int8 **)(v14 + 40);
  v128[0] = "omp_parallel";
  v16 = v13 - 2;
  v116 = v14;
  LOWORD(v129) = 259;
  sub_BD6B50(v15, v128);
  sub_D5F1F0((__int64)v12, v14);
  v117 = *(__int64 **)(a1 + 2704);
  v106 = sub_AD6530((__int64)v117, v14);
  v110 = *(_QWORD *)(a1 + 568);
  v122 = *(_WORD *)(a1 + 576);
  v119 = *(_QWORD *)(a1 + 560);
  v18 = sub_AA5190(a3);
  if ( v18 )
  {
    v19 = v17;
    v20 = HIBYTE(v17);
  }
  else
  {
    v20 = 0;
    v19 = 0;
  }
  LOBYTE(v21) = v19;
  HIBYTE(v21) = v20;
  sub_A88F30((__int64)v12, a3, v18, v21);
  v127 = 257;
  v115 = v16;
  v22 = sub_BCD420(v117, v16);
  v23 = sub_AA4E30(*(_QWORD *)(a1 + 560));
  v24 = sub_AE5260(v23, (__int64)v22);
  v25 = *(_DWORD *)(v23 + 4);
  v112 = v24;
  LOWORD(v129) = 257;
  v118 = (__int64)sub_BD2C40(80, 1u);
  if ( v118 )
    sub_B4CCA0(v118, v22, v25, 0, v112, (__int64)v128, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, __int64, __int64))(**(_QWORD **)(a1 + 600) + 16LL))(
    *(_QWORD *)(a1 + 600),
    v118,
    v126,
    v12[7],
    v12[8]);
  v26 = *(_QWORD *)(a1 + 512);
  if ( v26 != v26 + 16LL * *(unsigned int *)(a1 + 520) )
  {
    v27 = v26 + 16LL * *(unsigned int *)(a1 + 520);
    v113 = v16;
    v28 = *(_QWORD *)(a1 + 512);
    do
    {
      v29 = *(_QWORD *)(v28 + 8);
      v30 = *(_DWORD *)v28;
      v28 += 16;
      sub_B99FD0(v118, v30, v29);
    }
    while ( v27 != v28 );
    v16 = v113;
  }
  if ( *(_DWORD *)(*(_QWORD *)(v118 + 8) + 8LL) >> 8 )
  {
    v127 = 257;
    if ( v117 != *(__int64 **)(v118 + 8) )
    {
      if ( *(_BYTE *)v118 > 0x15u )
      {
        LOWORD(v129) = 257;
        v118 = sub_B52210(v118, (__int64)v117, (__int64)v128, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, __int64, __int64))(**(_QWORD **)(a1 + 600) + 16LL))(
          *(_QWORD *)(a1 + 600),
          v118,
          v126,
          v12[7],
          v12[8]);
        v94 = *(_QWORD *)(a1 + 512);
        v95 = 16LL * *(unsigned int *)(a1 + 520);
        v96 = v94 + v95;
        if ( v94 != v94 + v95 )
        {
          v114 = v16;
          v97 = *(_QWORD *)(a1 + 512);
          do
          {
            v98 = *(_QWORD *)(v97 + 8);
            v99 = *(_DWORD *)v97;
            v97 += 16;
            sub_B99FD0(v118, v99, v98);
          }
          while ( v96 != v97 );
          goto LABEL_57;
        }
      }
      else
      {
        v86 = *(_QWORD *)(a1 + 592);
        v87 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v86 + 136LL);
        if ( v87 == sub_928970 )
          v118 = sub_ADAFB0(v118, (__int64)v117);
        else
          v118 = v87(v86, v118, (__int64)v117);
        if ( *(_BYTE *)v118 > 0x1Cu )
        {
          (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, __int64, __int64))(**(_QWORD **)(a1 + 600) + 16LL))(
            *(_QWORD *)(a1 + 600),
            v118,
            v126,
            v12[7],
            v12[8]);
          v88 = *(_QWORD *)(a1 + 512);
          v89 = 16LL * *(unsigned int *)(a1 + 520);
          v90 = v88 + v89;
          if ( v88 != v88 + v89 )
          {
            v114 = v16;
            v91 = *(_QWORD *)(a1 + 512);
            do
            {
              v92 = *(_QWORD *)(v91 + 8);
              v93 = *(_DWORD *)v91;
              v91 += 16;
              sub_B99FD0(v118, v93, v92);
            }
            while ( v90 != v91 );
LABEL_57:
            v16 = v114;
          }
        }
      }
    }
  }
  if ( !v119 )
  {
    *(_QWORD *)(a1 + 560) = 0;
    *(_QWORD *)(a1 + 568) = 0;
    *(_WORD *)(a1 + 576) = 0;
    if ( v16 )
      goto LABEL_12;
LABEL_47:
    if ( a5 )
      goto LABEL_19;
LABEL_48:
    v85 = sub_BCB2D0(*(_QWORD **)(a1 + 584));
    a5 = sub_ACD640(v85, 1, 0);
    goto LABEL_29;
  }
  sub_A88F30((__int64)v12, v119, v110, v122);
  if ( !v16 )
    goto LABEL_47;
LABEL_12:
  v120 = v12;
  v31 = 0;
  v111 = v16;
  do
  {
    v32 = *(_QWORD *)(v116 + 32 * (v31 + 2LL - (*(_DWORD *)(v116 + 4) & 0x7FFFFFF)));
    LOWORD(v129) = 257;
    v33 = sub_BCD420(v117, v115);
    v34 = sub_3122740(v120, (__int64)v33, v118, 0, v31, (__int64)v128);
    v35 = sub_AA4E30(*(_QWORD *)(a1 + 560));
    v36 = sub_AE5020(v35, *(_QWORD *)(v32 + 8));
    HIBYTE(v37) = HIBYTE(v123);
    LOWORD(v129) = 257;
    LOBYTE(v37) = v36;
    v123 = v37;
    v38 = sub_BD2C40(80, unk_3F10A10);
    v40 = (__int64)v38;
    if ( v38 )
      sub_B4D3C0((__int64)v38, v32, v34, 0, v123, v39, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, const char **, __int64, __int64))(**(_QWORD **)(a1 + 600) + 16LL))(
      *(_QWORD *)(a1 + 600),
      v40,
      v128,
      v120[7],
      v120[8]);
    v41 = *(_QWORD *)(a1 + 512);
    v42 = v41 + 16LL * *(unsigned int *)(a1 + 520);
    while ( v42 != v41 )
    {
      v43 = *(_QWORD *)(v41 + 8);
      v44 = *(_DWORD *)v41;
      v41 += 16;
      sub_B99FD0(v40, v44, v43);
    }
    ++v31;
  }
  while ( v111 != v31 );
  v12 = v120;
  if ( !a5 )
    goto LABEL_48;
LABEL_19:
  v45 = *(__int64 ***)(a1 + 2632);
  v127 = 257;
  v46 = *(_QWORD *)(a5 + 8);
  v47 = sub_BCB060(v46);
  v48 = sub_BCB060((__int64)v45);
  if ( v47 < v48 )
  {
    a5 = sub_31223E0(v12, 0x28u, a5, v45, (__int64)v126, 0, (int)v128[0], 0);
    goto LABEL_29;
  }
  if ( v45 != (__int64 **)v46 && v47 != v48 )
  {
    v49 = *(_QWORD *)(a1 + 592);
    v50 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v49 + 120LL);
    if ( v50 == sub_920130 )
    {
      if ( *(_BYTE *)a5 > 0x15u )
      {
LABEL_64:
        LOWORD(v129) = 257;
        a5 = sub_B51D30(38, a5, (__int64)v45, (__int64)v128, 0, 0);
        (*(void (__fastcall **)(_QWORD, unsigned __int64, _BYTE *, __int64, __int64))(**(_QWORD **)(a1 + 600) + 16LL))(
          *(_QWORD *)(a1 + 600),
          a5,
          v126,
          v12[7],
          v12[8]);
        v101 = *(_QWORD *)(a1 + 512);
        v102 = v101 + 16LL * *(unsigned int *)(a1 + 520);
        while ( v102 != v101 )
        {
          v103 = *(_QWORD *)(v101 + 8);
          v104 = *(_DWORD *)v101;
          v101 += 16;
          sub_B99FD0(a5, v104, v103);
        }
        goto LABEL_29;
      }
      if ( (unsigned __int8)sub_AC4810(0x26u) )
        v51 = sub_ADAB70(38, a5, v45, 0);
      else
        v51 = sub_AA93C0(0x26u, a5, (__int64)v45);
    }
    else
    {
      v51 = v50(v49, 38u, (_BYTE *)a5, (__int64)v45);
    }
    if ( v51 )
    {
      a5 = v51;
      goto LABEL_29;
    }
    goto LABEL_64;
  }
LABEL_29:
  v128[0] = a4;
  v128[1] = a9;
  v128[2] = (const char *)a5;
  if ( !a6 )
  {
    v100 = sub_BCB2D0(*(_QWORD **)(a1 + 584));
    a6 = (const char *)sub_ACD640(v100, 0xFFFFFFFFLL, 0);
  }
  v52 = *(_QWORD **)(a1 + 584);
  v128[3] = a6;
  v53 = sub_BCB2D0(v52);
  v129 = sub_ACD640(v53, 0xFFFFFFFFLL, 0);
  v54 = *(__int64 ***)(a1 + 2928);
  v127 = 257;
  v55 = sub_31223E0(v12, 0x31u, a2, v54, (__int64)v126, 0, v124[0], 0);
  v56 = *(_QWORD **)(a1 + 584);
  v130 = v55;
  v131 = v106;
  v132 = v118;
  v57 = sub_BCB2E0(v56);
  v133 = sub_ACD640(v57, v115, 0);
  v58 = 0;
  v59 = sub_3135910(a1, 158);
  if ( v59 )
    v58 = *((_QWORD *)v59 + 3);
  v127 = 257;
  sub_921880((unsigned int **)v12, v58, (int)v59, (int)v128, 9, (__int64)v126, 0);
  sub_D5F1F0((__int64)v12, a7);
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    sub_B2C6D0(a2, a7, v60, v61);
  v62 = *(_QWORD *)(a1 + 560);
  v63 = *(_QWORD *)(a1 + 2632);
  v125 = 257;
  v64 = *(_QWORD *)(a2 + 96);
  v65 = sub_AA4E30(v62);
  v121 = sub_AE5020(v65, v63);
  v127 = 257;
  v66 = sub_BD2C40(80, 1u);
  v67 = (__int64)v66;
  if ( v66 )
    sub_B4D190((__int64)v66, v63, v64, (__int64)v126, 0, v121, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, int *, __int64, __int64))(**(_QWORD **)(a1 + 600) + 16LL))(
    *(_QWORD *)(a1 + 600),
    v67,
    v124,
    v12[7],
    v12[8]);
  v68 = *(_QWORD *)(a1 + 512);
  v69 = v68 + 16LL * *(unsigned int *)(a1 + 520);
  while ( v69 != v68 )
  {
    v70 = *(_QWORD *)(v68 + 8);
    v71 = *(_DWORD *)v68;
    v68 += 16;
    sub_B99FD0(v67, v71, v70);
  }
  v72 = sub_AA4E30(*(_QWORD *)(a1 + 560));
  v73 = sub_AE5020(v72, *(_QWORD *)(v67 + 8));
  v127 = 257;
  v74 = sub_BD2C40(80, unk_3F10A10);
  v76 = (__int64)v74;
  if ( v74 )
    sub_B4D3C0((__int64)v74, v67, a8, 0, v73, v75, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, __int64, __int64))(**(_QWORD **)(a1 + 600) + 16LL))(
    *(_QWORD *)(a1 + 600),
    v76,
    v126,
    v12[7],
    v12[8]);
  v77 = *(_QWORD *)(a1 + 512);
  v78 = v77 + 16LL * *(unsigned int *)(a1 + 520);
  while ( v78 != v77 )
  {
    v79 = *(_QWORD *)(v77 + 8);
    v80 = *(_DWORD *)v77;
    v77 += 16;
    sub_B99FD0(v76, v80, v79);
  }
  sub_B43D60((_QWORD *)v116);
  v81 = *(_QWORD ***)a10;
  result = *(unsigned int *)(a10 + 8);
  v83 = *(_QWORD *)a10 + 8 * result;
  if ( *(_QWORD *)a10 != v83 )
  {
    do
    {
      v84 = *v81++;
      result = sub_B43D60(v84);
    }
    while ( (_QWORD **)v83 != v81 );
  }
  return result;
}
