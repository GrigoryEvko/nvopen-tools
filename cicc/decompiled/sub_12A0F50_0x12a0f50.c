// Function: sub_12A0F50
// Address: 0x12a0f50
//
__int64 __fastcall sub_12A0F50(__int64 a1, __int64 a2)
{
  char *v4; // r14
  int v5; // r13d
  __int64 v6; // r15
  __int64 v7; // rax
  char v8; // dl
  _BYTE *v9; // rax
  int v10; // r9d
  __int64 v11; // r8
  int v12; // r11d
  int v13; // eax
  __int64 v14; // rax
  unsigned int v15; // esi
  __int64 v16; // r14
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 *v19; // r13
  __int64 v20; // rax
  _QWORD *v21; // rdi
  __int64 *v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // r13
  const char *v25; // r15
  __int64 v26; // rax
  __int64 v27; // rdi
  char v28; // dl
  __int64 v29; // rax
  unsigned __int64 v30; // r14
  int v31; // r12d
  __int64 v32; // rax
  __int64 v33; // r12
  int v34; // eax
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // r12
  __int64 v42; // rax
  int v43; // eax
  __int64 *v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r13
  __int64 *v48; // rdi
  __int64 v49; // rsi
  __int64 *v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rdx
  int v53; // r10d
  __int64 *v54; // r9
  int v55; // eax
  int v56; // edx
  int v57; // eax
  int v58; // r8d
  __int64 v59; // r9
  unsigned int v60; // eax
  __int64 v61; // rdi
  int v62; // esi
  __int64 *v63; // rcx
  int v64; // eax
  int v65; // esi
  __int64 v66; // rdi
  __int64 *v67; // r8
  unsigned int v68; // r10d
  int v69; // eax
  __int64 v70; // rcx
  int v71; // [rsp+10h] [rbp-130h]
  int v72; // [rsp+30h] [rbp-110h]
  int v73; // [rsp+30h] [rbp-110h]
  int v74; // [rsp+38h] [rbp-108h]
  __int64 v75; // [rsp+38h] [rbp-108h]
  __int64 v76; // [rsp+40h] [rbp-100h]
  __int64 v77; // [rsp+40h] [rbp-100h]
  int v78; // [rsp+48h] [rbp-F8h]
  __int64 v79; // [rsp+48h] [rbp-F8h]
  unsigned int v80; // [rsp+48h] [rbp-F8h]
  int v81; // [rsp+50h] [rbp-F0h] BYREF
  int v82; // [rsp+54h] [rbp-ECh] BYREF
  __int64 v83; // [rsp+58h] [rbp-E8h] BYREF
  char *s; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v85; // [rsp+70h] [rbp-D0h] BYREF
  _BYTE *v86; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v87; // [rsp+88h] [rbp-B8h]
  _BYTE v88[176]; // [rsp+90h] [rbp-B0h] BYREF

  v76 = 8LL * *(_QWORD *)(a2 + 128);
  if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
    v78 = 8 * sub_8D4AB0(a2);
  else
    v78 = 8 * *(_DWORD *)(a2 + 136);
  sub_129EFF0(&s, a1, a2);
  v4 = s;
  sub_129E300(*(_DWORD *)(a2 + 64), (char *)&v81);
  v5 = sub_129F850(a1, *(_DWORD *)(a2 + 64));
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 140) - 10) > 1u )
    sub_127B550("unexpected: aggregate type not struct/union!", (_DWORD *)(a2 + 64), 1);
  v6 = a1 + 16;
  if ( (*(_BYTE *)(a2 + 141) & 8) == 0 )
  {
    v7 = *(_QWORD *)(a2 + 40);
    v8 = *(_BYTE *)(a2 + 89) & 2;
    if ( v7 )
    {
      v9 = (_BYTE *)(v7 - 8);
      if ( !v8 )
        goto LABEL_9;
      goto LABEL_65;
    }
    if ( v8 )
    {
LABEL_65:
      v9 = (_BYTE *)(sub_72F070(a2) - 8);
LABEL_9:
      v10 = v81;
      LODWORD(v11) = v5;
      if ( (*v9 & 1) != 0 )
        goto LABEL_10;
      goto LABEL_54;
    }
  }
  v10 = v81;
LABEL_54:
  v42 = *(_QWORD *)(a1 + 544);
  LODWORD(v11) = v5;
  if ( v42 != *(_QWORD *)(a1 + 512) )
  {
    if ( v42 == *(_QWORD *)(a1 + 552) )
      v42 = *(_QWORD *)(*(_QWORD *)(a1 + 568) - 8LL) + 512LL;
    v11 = *(_QWORD *)(v42 - 8);
  }
LABEL_10:
  v12 = 0;
  if ( v4 )
  {
    v72 = v10;
    v74 = v11;
    v13 = strlen(v4);
    v10 = v72;
    LODWORD(v11) = v74;
    v12 = v13;
  }
  v83 = sub_15A6F30(
          (int)a1 + 16,
          4 * (unsigned int)(*(_BYTE *)(a2 + 140) != 10) + 19,
          (_DWORD)v4,
          v12,
          v11,
          v5,
          v10,
          0,
          v76,
          v78,
          0,
          (__int64)byte_3F871B3,
          0);
  v14 = sub_1621440(v83);
  v15 = *(_DWORD *)(a1 + 632);
  v83 = v14;
  v16 = v14;
  if ( !v15 )
  {
    ++*(_QWORD *)(a1 + 608);
    goto LABEL_84;
  }
  v17 = *(_QWORD *)(a1 + 616);
  v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v19 = (__int64 *)(v17 + 16 * v18);
  v20 = *v19;
  if ( *v19 == a2 )
  {
LABEL_14:
    v21 = v19 + 1;
    if ( v19[1] )
    {
      sub_161E7C0(v21);
      v21 = v19 + 1;
    }
    goto LABEL_16;
  }
  v53 = 1;
  v54 = 0;
  while ( v20 != -8 )
  {
    if ( !v54 && v20 == -16 )
      v54 = v19;
    LODWORD(v18) = (v15 - 1) & (v53 + v18);
    v19 = (__int64 *)(v17 + 16LL * (unsigned int)v18);
    v20 = *v19;
    if ( *v19 == a2 )
      goto LABEL_14;
    ++v53;
  }
  v55 = *(_DWORD *)(a1 + 624);
  if ( v54 )
    v19 = v54;
  ++*(_QWORD *)(a1 + 608);
  v56 = v55 + 1;
  if ( 4 * (v55 + 1) >= 3 * v15 )
  {
LABEL_84:
    sub_12A0A00(a1 + 608, 2 * v15);
    v57 = *(_DWORD *)(a1 + 632);
    if ( v57 )
    {
      v58 = v57 - 1;
      v59 = *(_QWORD *)(a1 + 616);
      v60 = (v57 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v56 = *(_DWORD *)(a1 + 624) + 1;
      v19 = (__int64 *)(v59 + 16LL * v60);
      v61 = *v19;
      if ( *v19 != a2 )
      {
        v62 = 1;
        v63 = 0;
        while ( v61 != -8 )
        {
          if ( !v63 && v61 == -16 )
            v63 = v19;
          v60 = v58 & (v62 + v60);
          v19 = (__int64 *)(v59 + 16LL * v60);
          v61 = *v19;
          if ( *v19 == a2 )
            goto LABEL_80;
          ++v62;
        }
        if ( v63 )
          v19 = v63;
      }
      goto LABEL_80;
    }
    goto LABEL_113;
  }
  if ( v15 - *(_DWORD *)(a1 + 628) - v56 <= v15 >> 3 )
  {
    v80 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    sub_12A0A00(a1 + 608, v15);
    v64 = *(_DWORD *)(a1 + 632);
    if ( v64 )
    {
      v65 = v64 - 1;
      v66 = *(_QWORD *)(a1 + 616);
      v67 = 0;
      v68 = v65 & v80;
      v56 = *(_DWORD *)(a1 + 624) + 1;
      v69 = 1;
      v19 = (__int64 *)(v66 + 16LL * (v65 & v80));
      v70 = *v19;
      if ( *v19 != a2 )
      {
        while ( v70 != -8 )
        {
          if ( !v67 && v70 == -16 )
            v67 = v19;
          v68 = v65 & (v69 + v68);
          v19 = (__int64 *)(v66 + 16LL * v68);
          v70 = *v19;
          if ( *v19 == a2 )
            goto LABEL_80;
          ++v69;
        }
        if ( v67 )
          v19 = v67;
      }
      goto LABEL_80;
    }
LABEL_113:
    ++*(_DWORD *)(a1 + 624);
    BUG();
  }
LABEL_80:
  *(_DWORD *)(a1 + 624) = v56;
  if ( *v19 != -8 )
    --*(_DWORD *)(a1 + 628);
  *v19 = a2;
  v21 = v19 + 1;
  v16 = v83;
  v19[1] = 0;
LABEL_16:
  v19[1] = v16;
  if ( v16 )
    sub_1623A60(v21, v16, 2);
  v22 = *(__int64 **)(a1 + 544);
  if ( v22 == (__int64 *)(*(_QWORD *)(a1 + 560) - 8LL) )
  {
    v47 = *(_QWORD *)(a1 + 568);
    if ( ((((v47 - *(_QWORD *)(a1 + 536)) >> 3) - 1) << 6)
       + (((__int64)v22 - *(_QWORD *)(a1 + 552)) >> 3)
       + ((__int64)(*(_QWORD *)(a1 + 528) - *(_QWORD *)(a1 + 512)) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 504) - ((v47 - *(_QWORD *)(a1 + 496)) >> 3)) <= 1 )
    {
      sub_129F230((__int64 *)(a1 + 496), 1u, 0);
      v47 = *(_QWORD *)(a1 + 568);
    }
    *(_QWORD *)(v47 + 8) = sub_22077B0(512);
    v48 = *(__int64 **)(a1 + 544);
    if ( v48 )
    {
      v49 = v83;
      *v48 = v83;
      if ( v49 )
        sub_1623A60(v48, v49, 2);
    }
    v50 = (__int64 *)(*(_QWORD *)(a1 + 568) + 8LL);
    *(_QWORD *)(a1 + 568) = v50;
    v51 = *v50;
    v52 = *v50 + 512;
    *(_QWORD *)(a1 + 552) = v51;
    *(_QWORD *)(a1 + 560) = v52;
    *(_QWORD *)(a1 + 544) = v51;
  }
  else
  {
    if ( v22 )
    {
      v23 = v83;
      *v22 = v83;
      if ( v23 )
        sub_1623A60(v22, v23, 2);
      v22 = *(__int64 **)(a1 + 544);
    }
    *(_QWORD *)(a1 + 544) = v22 + 1;
  }
  v24 = *(_QWORD *)(a2 + 160);
  v86 = v88;
  v87 = 0x1000000000LL;
  if ( v24 )
  {
    v73 = a1 + 16;
    while ( 1 )
    {
      if ( (*(_BYTE *)(v24 + 89) & 8) != 0 && (v25 = *(const char **)(v24 + 24)) != 0
        || (v25 = *(const char **)(v24 + 8)) != 0 )
      {
        if ( memcmp(v25, "__nv_no_debug_dummy", 0x13u) )
          goto LABEL_32;
        v24 = *(_QWORD *)(v24 + 112);
        if ( !v24 )
          goto LABEL_44;
      }
      else
      {
        v25 = byte_3F871B3;
LABEL_32:
        v26 = sub_12A0C10(a1, *(_QWORD *)(v24 + 120));
        v27 = *(_QWORD *)(v24 + 120);
        v77 = v26;
        v28 = *(_BYTE *)(v27 + 140);
        v29 = v27;
        if ( v28 == 12 )
        {
          do
            v29 = *(_QWORD *)(v29 + 160);
          while ( *(_BYTE *)(v29 + 140) == 12 );
        }
        v30 = *(_QWORD *)(v29 + 128);
        v79 = 8 * v30;
        if ( (*(_BYTE *)(v24 + 144) & 4) != 0 )
          v79 = *(unsigned __int8 *)(v24 + 137);
        if ( *(char *)(v27 + 142) >= 0 && v28 == 12 )
          v31 = 8 * sub_8D4AB0(v27);
        else
          v31 = 8 * *(_DWORD *)(v27 + 136);
        v32 = *(unsigned __int8 *)(v24 + 136);
        if ( (_BYTE)v32 && (*(_BYTE *)(v24 + 144) & 4) == 0 )
          sub_127B550("invalid bit offset of struct field!", (_DWORD *)(v24 + 64), 1);
        v75 = v32 + 8LL * *(_QWORD *)(v24 + 128);
        sub_129E300(*(_DWORD *)(v24 + 64), (char *)&v82);
        v71 = sub_129F850(a1, *(_DWORD *)(v24 + 64));
        if ( (*(_BYTE *)(v24 + 144) & 4) != 0 )
        {
          v33 = 8 * v30 * (*(_QWORD *)(v24 + 128) / v30);
          v34 = strlen(v25);
          v35 = sub_15A5CB0(v73, v83, (_DWORD)v25, v34, v71, v82, v79, v75, v33, 0, v77);
          v36 = (unsigned int)v87;
          if ( (unsigned int)v87 >= HIDWORD(v87) )
            goto LABEL_59;
        }
        else
        {
          v43 = strlen(v25);
          v35 = sub_15A5C20(v73, v83, (_DWORD)v25, v43, v71, v82, v79, v31, v75, 0, v77);
          v36 = (unsigned int)v87;
          if ( (unsigned int)v87 >= HIDWORD(v87) )
          {
LABEL_59:
            sub_16CD150(&v86, v88, 0, 8);
            v36 = (unsigned int)v87;
          }
        }
        *(_QWORD *)&v86[8 * v36] = v35;
        v24 = *(_QWORD *)(v24 + 112);
        LODWORD(v87) = v87 + 1;
        if ( !v24 )
        {
LABEL_44:
          v6 = a1 + 16;
          break;
        }
      }
    }
  }
  v37 = *(_QWORD *)(a1 + 544);
  if ( v37 == *(_QWORD *)(a1 + 552) )
  {
    j_j___libc_free_0(v37, 512);
    v44 = (__int64 *)(*(_QWORD *)(a1 + 568) - 8LL);
    *(_QWORD *)(a1 + 568) = v44;
    v45 = *v44;
    v46 = *v44 + 512;
    *(_QWORD *)(a1 + 552) = v45;
    *(_QWORD *)(a1 + 560) = v46;
    *(_QWORD *)(a1 + 544) = v45 + 504;
    if ( *(_QWORD *)(v45 + 504) )
      sub_161E7C0(v45 + 504);
  }
  else
  {
    *(_QWORD *)(a1 + 544) = v37 - 8;
    if ( *(_QWORD *)(v37 - 8) )
      sub_161E7C0(v37 - 8);
  }
  v38 = sub_15A5DC0(v6, v86, (unsigned int)v87);
  sub_15A7250(v6, &v83, v38, 0);
  v39 = v83;
  sub_15A7340(v6, v83);
  v40 = v83;
  if ( v86 != v88 )
    _libc_free(v86, v39);
  if ( s != (char *)&v85 )
    j_j___libc_free_0(s, v85 + 1);
  return v40;
}
