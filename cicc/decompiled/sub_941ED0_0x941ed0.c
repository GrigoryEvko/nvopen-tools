// Function: sub_941ED0
// Address: 0x941ed0
//
__int64 __fastcall sub_941ED0(__int64 a1, __int64 a2)
{
  char *v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  int v8; // r9d
  int v9; // r12d
  __int64 v10; // r15
  __int64 v11; // rax
  char v12; // dl
  char *v13; // rax
  char v14; // al
  int v15; // edx
  __int64 v16; // r8
  int v17; // ecx
  int v18; // eax
  __int64 v19; // rax
  unsigned int v20; // esi
  __int64 v21; // r12
  __int64 v22; // r8
  int v23; // r11d
  __int64 *v24; // rdi
  unsigned int v25; // ecx
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // r13
  __int64 *v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // r12
  const char *v32; // r15
  __int64 v33; // rax
  __int64 v34; // rdi
  char v35; // dl
  __int64 v36; // rax
  unsigned __int64 v37; // r13
  int v38; // ebx
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  int v43; // r9d
  __int64 v44; // rbx
  int v45; // eax
  __int64 v46; // rbx
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // r12
  __int64 v54; // rax
  int v55; // eax
  __int64 *v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r12
  __int64 *v60; // rdi
  __int64 v61; // rsi
  __int64 *v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  int v65; // eax
  int v66; // edx
  int v67; // eax
  int v68; // r9d
  __int64 v69; // r10
  unsigned int v70; // eax
  __int64 v71; // r8
  int v72; // esi
  __int64 *v73; // rcx
  int v74; // eax
  int v75; // esi
  __int64 v76; // r8
  __int64 *v77; // r9
  unsigned int v78; // r13d
  int v79; // eax
  __int64 v80; // rcx
  int v81; // [rsp+10h] [rbp-130h]
  int v82; // [rsp+30h] [rbp-110h]
  int v83; // [rsp+30h] [rbp-110h]
  int v84; // [rsp+38h] [rbp-108h]
  __int64 v85; // [rsp+38h] [rbp-108h]
  __int64 v86; // [rsp+40h] [rbp-100h]
  __int64 v87; // [rsp+40h] [rbp-100h]
  int v88; // [rsp+48h] [rbp-F8h]
  __int64 v89; // [rsp+48h] [rbp-F8h]
  int v90; // [rsp+50h] [rbp-F0h] BYREF
  int v91; // [rsp+54h] [rbp-ECh] BYREF
  __int64 v92; // [rsp+58h] [rbp-E8h] BYREF
  char *s; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v94; // [rsp+70h] [rbp-D0h] BYREF
  _BYTE *v95; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v96; // [rsp+88h] [rbp-B8h]
  _BYTE v97[176]; // [rsp+90h] [rbp-B0h] BYREF

  v86 = 8LL * *(_QWORD *)(a2 + 128);
  if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
    v88 = 8 * sub_8D4AB0(a2);
  else
    v88 = 8 * *(_DWORD *)(a2 + 136);
  sub_93FAB0(&s, a1, a2);
  v4 = s;
  sub_93ED80(*(_DWORD *)(a2 + 64), (char *)&v90);
  v9 = sub_9405D0(a1, *(_DWORD *)(a2 + 64), v5, v6, v7, v8);
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 140) - 10) > 1u )
    sub_91B8A0("unexpected: aggregate type not struct/union!", (_DWORD *)(a2 + 64), 1);
  v10 = a1 + 16;
  if ( (*(_BYTE *)(a2 + 141) & 8) == 0 )
  {
    v11 = *(_QWORD *)(a2 + 40);
    v12 = *(_BYTE *)(a2 + 89) & 2;
    if ( v11 )
    {
      v13 = (char *)(v11 - 8);
      if ( !v12 )
        goto LABEL_9;
      goto LABEL_66;
    }
    if ( v12 )
    {
LABEL_66:
      v13 = (char *)(sub_72F070(a2) - 8);
LABEL_9:
      v14 = *v13;
      v15 = v90;
      BYTE4(v95) = 0;
      LODWORD(v16) = v9;
      if ( (v14 & 1) != 0 )
        goto LABEL_10;
      goto LABEL_56;
    }
  }
  BYTE4(v95) = 0;
  v15 = v90;
LABEL_56:
  v54 = *(_QWORD *)(a1 + 512);
  LODWORD(v16) = v9;
  if ( v54 != *(_QWORD *)(a1 + 480) )
  {
    if ( v54 == *(_QWORD *)(a1 + 520) )
      v54 = *(_QWORD *)(*(_QWORD *)(a1 + 536) - 8LL) + 512LL;
    v16 = *(_QWORD *)(v54 - 8);
  }
LABEL_10:
  v17 = 0;
  if ( v4 )
  {
    v82 = v15;
    v84 = v16;
    v18 = strlen(v4);
    v15 = v82;
    v17 = v18;
    LODWORD(v16) = v84;
  }
  v92 = sub_ADE2D0(
          (int)a1 + 16,
          4 * (unsigned int)(*(_BYTE *)(a2 + 140) != 10) + 19,
          (_DWORD)v4,
          v17,
          v16,
          v9,
          v15,
          0,
          v86,
          v88,
          0,
          (__int64)byte_3F871B3,
          0,
          0,
          (__int64)v95);
  v19 = sub_B95B00(v92);
  v20 = *(_DWORD *)(a1 + 600);
  v92 = v19;
  v21 = v19;
  if ( !v20 )
  {
    ++*(_QWORD *)(a1 + 576);
    goto LABEL_89;
  }
  v22 = *(_QWORD *)(a1 + 584);
  v23 = 1;
  v24 = 0;
  v25 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v26 = (__int64 *)(v22 + 16LL * v25);
  v27 = *v26;
  if ( *v26 == a2 )
  {
LABEL_14:
    v28 = v26 + 1;
    if ( v26[1] )
      sub_B91220(v26 + 1);
    goto LABEL_16;
  }
  while ( v27 != -4096 )
  {
    if ( !v24 && v27 == -8192 )
      v24 = v26;
    v25 = (v20 - 1) & (v23 + v25);
    v26 = (__int64 *)(v22 + 16LL * v25);
    v27 = *v26;
    if ( *v26 == a2 )
      goto LABEL_14;
    ++v23;
  }
  if ( !v24 )
    v24 = v26;
  v65 = *(_DWORD *)(a1 + 592);
  ++*(_QWORD *)(a1 + 576);
  v66 = v65 + 1;
  if ( 4 * (v65 + 1) >= 3 * v20 )
  {
LABEL_89:
    sub_941970(a1 + 576, 2 * v20);
    v67 = *(_DWORD *)(a1 + 600);
    if ( v67 )
    {
      v68 = v67 - 1;
      v69 = *(_QWORD *)(a1 + 584);
      v70 = (v67 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v66 = *(_DWORD *)(a1 + 592) + 1;
      v24 = (__int64 *)(v69 + 16LL * v70);
      v71 = *v24;
      if ( *v24 != a2 )
      {
        v72 = 1;
        v73 = 0;
        while ( v71 != -4096 )
        {
          if ( !v73 && v71 == -8192 )
            v73 = v24;
          v70 = v68 & (v72 + v70);
          v24 = (__int64 *)(v69 + 16LL * v70);
          v71 = *v24;
          if ( *v24 == a2 )
            goto LABEL_85;
          ++v72;
        }
        if ( v73 )
          v24 = v73;
      }
      goto LABEL_85;
    }
    goto LABEL_113;
  }
  if ( v20 - *(_DWORD *)(a1 + 596) - v66 <= v20 >> 3 )
  {
    sub_941970(a1 + 576, v20);
    v74 = *(_DWORD *)(a1 + 600);
    if ( v74 )
    {
      v75 = v74 - 1;
      v76 = *(_QWORD *)(a1 + 584);
      v77 = 0;
      v78 = (v74 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v66 = *(_DWORD *)(a1 + 592) + 1;
      v79 = 1;
      v24 = (__int64 *)(v76 + 16LL * v78);
      v80 = *v24;
      if ( *v24 != a2 )
      {
        while ( v80 != -4096 )
        {
          if ( v80 == -8192 && !v77 )
            v77 = v24;
          v78 = v75 & (v79 + v78);
          v24 = (__int64 *)(v76 + 16LL * v78);
          v80 = *v24;
          if ( *v24 == a2 )
            goto LABEL_85;
          ++v79;
        }
        if ( v77 )
          v24 = v77;
      }
      goto LABEL_85;
    }
LABEL_113:
    ++*(_DWORD *)(a1 + 592);
    BUG();
  }
LABEL_85:
  *(_DWORD *)(a1 + 592) = v66;
  if ( *v24 != -4096 )
    --*(_DWORD *)(a1 + 596);
  *v24 = a2;
  v28 = v24 + 1;
  v21 = v92;
  v24[1] = 0;
LABEL_16:
  *v28 = v21;
  if ( v21 )
    sub_B96E90(v28, v21, 1);
  v29 = *(__int64 **)(a1 + 512);
  if ( v29 == (__int64 *)(*(_QWORD *)(a1 + 528) - 8LL) )
  {
    v59 = *(_QWORD *)(a1 + 536);
    if ( ((((v59 - *(_QWORD *)(a1 + 504)) >> 3) - 1) << 6)
       + (((__int64)v29 - *(_QWORD *)(a1 + 520)) >> 3)
       + ((__int64)(*(_QWORD *)(a1 + 496) - *(_QWORD *)(a1 + 480)) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 472) - ((v59 - *(_QWORD *)(a1 + 464)) >> 3)) <= 1 )
    {
      sub_93FFB0((__int64 *)(a1 + 464), 1u, 0);
      v59 = *(_QWORD *)(a1 + 536);
    }
    *(_QWORD *)(v59 + 8) = sub_22077B0(512);
    v60 = *(__int64 **)(a1 + 512);
    if ( v60 )
    {
      v61 = v92;
      *v60 = v92;
      if ( v61 )
        sub_B96E90(v60, v61, 1);
    }
    v62 = (__int64 *)(*(_QWORD *)(a1 + 536) + 8LL);
    *(_QWORD *)(a1 + 536) = v62;
    v63 = *v62;
    v64 = *v62 + 512;
    *(_QWORD *)(a1 + 520) = v63;
    *(_QWORD *)(a1 + 528) = v64;
    *(_QWORD *)(a1 + 512) = v63;
  }
  else
  {
    if ( v29 )
    {
      v30 = v92;
      *v29 = v92;
      if ( v30 )
        sub_B96E90(v29, v30, 1);
      v29 = *(__int64 **)(a1 + 512);
    }
    *(_QWORD *)(a1 + 512) = v29 + 1;
  }
  v31 = *(_QWORD *)(a2 + 160);
  v95 = v97;
  v96 = 0x1000000000LL;
  if ( v31 )
  {
    v83 = a1 + 16;
    while ( 1 )
    {
      if ( (*(_BYTE *)(v31 + 89) & 8) == 0 || (v32 = *(const char **)(v31 + 24)) == 0 )
      {
        v32 = *(const char **)(v31 + 8);
        if ( !v32 )
          break;
      }
      if ( !memcmp(v32, "__nv_no_debug_dummy", 0x13u) )
      {
        v31 = *(_QWORD *)(v31 + 112);
        if ( !v31 )
          goto LABEL_46;
      }
      else
      {
LABEL_32:
        v33 = sub_941B90(a1, *(_QWORD *)(v31 + 120));
        v34 = *(_QWORD *)(v31 + 120);
        v87 = v33;
        v35 = *(_BYTE *)(v34 + 140);
        v36 = v34;
        if ( v35 == 12 )
        {
          do
            v36 = *(_QWORD *)(v36 + 160);
          while ( *(_BYTE *)(v36 + 140) == 12 );
        }
        v37 = *(_QWORD *)(v36 + 128);
        v89 = 8 * v37;
        if ( (*(_BYTE *)(v31 + 144) & 4) != 0 )
          v89 = *(unsigned __int8 *)(v31 + 137);
        if ( *(char *)(v34 + 142) >= 0 && v35 == 12 )
          v38 = 8 * sub_8D4AB0(v34);
        else
          v38 = 8 * *(_DWORD *)(v34 + 136);
        v39 = *(unsigned __int8 *)(v31 + 136);
        if ( (_BYTE)v39 && (*(_BYTE *)(v31 + 144) & 4) == 0 )
          sub_91B8A0("invalid bit offset of struct field!", (_DWORD *)(v31 + 64), 1);
        v85 = v39 + 8LL * *(_QWORD *)(v31 + 128);
        sub_93ED80(*(_DWORD *)(v31 + 64), (char *)&v91);
        v81 = sub_9405D0(a1, *(_DWORD *)(v31 + 64), v40, v41, v42, v43);
        if ( (*(_BYTE *)(v31 + 144) & 4) != 0 )
        {
          v44 = 8 * v37 * (*(_QWORD *)(v31 + 128) / v37);
          v45 = strlen(v32);
          v46 = sub_ADCC50(v83, v92, (_DWORD)v32, v45, v81, v91, v89, v85, v44, 0, v87, 0);
        }
        else
        {
          v55 = strlen(v32);
          v46 = sub_ADCBB0(v83, v92, (_DWORD)v32, v55, v81, v91, v89, v38, v85, 0, v87, 0);
        }
        v47 = (unsigned int)v96;
        v48 = (unsigned int)v96 + 1LL;
        if ( v48 > HIDWORD(v96) )
        {
          sub_C8D5F0(&v95, v97, v48, 8);
          v47 = (unsigned int)v96;
        }
        *(_QWORD *)&v95[8 * v47] = v46;
        LODWORD(v96) = v96 + 1;
        v31 = *(_QWORD *)(v31 + 112);
        if ( !v31 )
        {
LABEL_46:
          v10 = a1 + 16;
          goto LABEL_47;
        }
      }
    }
    v32 = byte_3F871B3;
    goto LABEL_32;
  }
LABEL_47:
  v49 = *(_QWORD *)(a1 + 512);
  if ( v49 == *(_QWORD *)(a1 + 520) )
  {
    j_j___libc_free_0(v49, 512);
    v56 = (__int64 *)(*(_QWORD *)(a1 + 536) - 8LL);
    *(_QWORD *)(a1 + 536) = v56;
    v57 = *v56;
    v58 = *v56 + 512;
    *(_QWORD *)(a1 + 520) = v57;
    *(_QWORD *)(a1 + 528) = v58;
    *(_QWORD *)(a1 + 512) = v57 + 504;
    if ( *(_QWORD *)(v57 + 504) )
      sub_B91220(v57 + 504);
  }
  else
  {
    *(_QWORD *)(a1 + 512) = v49 - 8;
    if ( *(_QWORD *)(v49 - 8) )
      sub_B91220(v49 - 8);
  }
  v50 = sub_ADCD70(v10, v95, (unsigned int)v96);
  sub_ADEAE0(v10, &v92, v50, 0);
  v51 = v92;
  sub_ADDCE0(v10, v92);
  v52 = v92;
  if ( v95 != v97 )
    _libc_free(v95, v51);
  if ( s != (char *)&v94 )
    j_j___libc_free_0(s, v94 + 1);
  return v52;
}
