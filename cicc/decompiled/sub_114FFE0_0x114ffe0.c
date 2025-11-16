// Function: sub_114FFE0
// Address: 0x114ffe0
//
__int64 __fastcall sub_114FFE0(__int64 a1, __int64 a2)
{
  unsigned __int16 v2; // ax
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned __int64 v5; // rdi
  int v6; // eax
  __int64 v7; // rdi
  unsigned __int8 v8; // r13
  unsigned int v9; // r13d
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r14
  _QWORD *v14; // rax
  int v15; // ecx
  _QWORD *v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r15
  _QWORD *v20; // rbx
  char *v21; // r13
  unsigned int v22; // r8d
  __int64 v23; // rbx
  char *v24; // r15
  __int64 i; // rbx
  unsigned __int8 *v26; // r13
  __int64 v27; // rcx
  __int64 v28; // r12
  _QWORD *j; // rbx
  __int64 v30; // r13
  unsigned int v31; // r8d
  __int64 v32; // rbx
  __int64 v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 v38; // rcx
  __int64 v39; // rdx
  int v40; // eax
  int v41; // eax
  unsigned int v42; // esi
  __int64 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rsi
  __int64 *v46; // rdi
  __int64 v47; // rdx
  int v48; // eax
  int v49; // eax
  unsigned int v50; // ecx
  __int64 v51; // rax
  __int64 v52; // rcx
  __int64 v53; // rcx
  _QWORD *v54; // rax
  const char **v55; // r8
  __int16 v56; // dx
  __int64 v57; // rbx
  char v58; // cl
  char v59; // dl
  unsigned __int64 v60; // rax
  unsigned __int16 v61; // r9
  _QWORD *v62; // rax
  _QWORD *v63; // r14
  __int64 v64; // rdx
  const char **v65; // rbx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rsi
  unsigned __int8 *v70; // rsi
  __int64 v71; // rsi
  unsigned __int8 *v72; // rsi
  __int64 v73; // [rsp+8h] [rbp-108h]
  char v74; // [rsp+14h] [rbp-FCh]
  char v75; // [rsp+18h] [rbp-F8h]
  __int64 v76; // [rsp+18h] [rbp-F8h]
  __int64 v77; // [rsp+20h] [rbp-F0h]
  __int16 v78; // [rsp+20h] [rbp-F0h]
  const char **v79; // [rsp+20h] [rbp-F0h]
  __int64 v80; // [rsp+20h] [rbp-F0h]
  __int64 v81; // [rsp+28h] [rbp-E8h]
  __int64 v82; // [rsp+28h] [rbp-E8h]
  unsigned __int8 v83; // [rsp+30h] [rbp-E0h]
  __int64 v84; // [rsp+30h] [rbp-E0h]
  __int64 v85; // [rsp+38h] [rbp-D8h]
  __int64 v86; // [rsp+38h] [rbp-D8h]
  unsigned __int8 v87; // [rsp+38h] [rbp-D8h]
  char v88; // [rsp+38h] [rbp-D8h]
  __int64 v89; // [rsp+40h] [rbp-D0h]
  __int64 v90; // [rsp+40h] [rbp-D0h]
  const char *v92; // [rsp+68h] [rbp-A8h] BYREF
  __int64 v93[4]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v94[4]; // [rsp+90h] [rbp-80h] BYREF
  const char *v95[4]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v96; // [rsp+D0h] [rbp-40h]

  v2 = *(_WORD *)(a2 + 2);
  if ( ((v2 >> 7) & 6) != 0 || (v2 & 1) != 0 )
    return 0;
  v3 = *(_QWORD *)(a2 + 40);
  v4 = a2;
  v5 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 == v3 + 48 )
  {
    v7 = 0;
  }
  else
  {
    if ( !v5 )
      BUG();
    v6 = *(unsigned __int8 *)(v5 - 24);
    v7 = v5 - 24;
    if ( (unsigned int)(v6 - 30) >= 0xB )
      v7 = 0;
  }
  v89 = sub_B46EC0(v7, 0);
  v8 = sub_AA5590(v89, 2);
  if ( !v8 )
    return 0;
  v11 = *(_QWORD *)(v89 + 16);
  if ( !v11 )
LABEL_121:
    BUG();
  while ( 1 )
  {
    v12 = *(_QWORD *)(v11 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v12 - 30) <= 0xAu )
      break;
    v11 = *(_QWORD *)(v11 + 8);
    if ( !v11 )
      goto LABEL_121;
  }
  v13 = *(_QWORD *)(v12 + 40);
  if ( v13 == v3 )
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        break;
      v27 = *(_QWORD *)(v11 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v27 - 30) <= 0xAu )
      {
        v13 = *(_QWORD *)(v27 + 40);
        goto LABEL_13;
      }
    }
    v13 = *(_QWORD *)(MEMORY[0x18] + 40LL);
  }
LABEL_13:
  if ( v89 == v3 || v89 == v13 )
    return 0;
  v14 = (_QWORD *)(*(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v14 == (_QWORD *)(v13 + 48) )
    goto LABEL_120;
  if ( !v14 )
    BUG();
  v15 = *((unsigned __int8 *)v14 - 24);
  if ( (unsigned int)(v15 - 30) > 0xA )
LABEL_120:
    BUG();
  if ( (_BYTE)v15 != 31 )
    return 0;
  v16 = *(_QWORD **)(v13 + 56);
  if ( v16 == v14 )
    return 0;
  if ( (*((_DWORD *)v14 - 5) & 0x7FFFFFF) != 1 )
  {
    v17 = *(v14 - 7);
    if ( v3 == v17 && v17 || (v18 = *(v14 - 11), v3 == v18) && v18 )
    {
      v85 = v3;
      v19 = 0;
      v20 = (_QWORD *)(*(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL);
      v83 = v8;
      while ( 1 )
      {
        v21 = (char *)(v20 - 3);
        if ( *((_BYTE *)v20 - 24) == 62
          && *(v20 - 7) == *(_QWORD *)(v4 - 32)
          && (unsigned __int8)sub_B50C50(
                                *(_QWORD *)(*(v20 - 11) + 8LL),
                                *(_QWORD *)(*(_QWORD *)(v4 - 64) + 8LL),
                                *(_QWORD *)(a1 + 88))
          && (unsigned __int8)sub_B45D20(v4, (__int64)(v20 - 3), 0, 0, v22) )
        {
          v23 = v85;
          v86 = v19;
          v24 = v21;
          for ( i = *(_QWORD *)(v23 + 56); ; i = *(_QWORD *)(i + 8) )
          {
            if ( i )
            {
              v26 = (unsigned __int8 *)(i - 24);
              if ( i - 24 == v4 )
              {
                v9 = v83;
                goto LABEL_61;
              }
            }
            else
            {
              v26 = 0;
            }
            if ( (unsigned __int8)sub_B46420((__int64)v26)
              || (unsigned __int8)sub_B46790(v26, 0)
              || (unsigned __int8)sub_B46490((__int64)v26) )
            {
              return 0;
            }
          }
        }
        if ( (unsigned __int8)sub_B46420((__int64)(v20 - 3))
          || (unsigned __int8)sub_B46790((unsigned __int8 *)v20 - 24, 0)
          || (unsigned __int8)sub_B46490((__int64)(v20 - 3))
          || *(_QWORD **)(v13 + 56) == v20 )
        {
          break;
        }
        LOWORD(v19) = 0;
        v20 = (_QWORD *)(*v20 & 0xFFFFFFFFFFFFFFF8LL);
        if ( !v20 )
          BUG();
      }
    }
    return 0;
  }
  v84 = v4;
  v28 = 0;
  v87 = v8;
  for ( j = (_QWORD *)(*v14 & 0xFFFFFFFFFFFFFFF8LL); ; j = (_QWORD *)(*j & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v30 = 0;
    if ( j )
      v30 = (__int64)(j - 3);
    if ( !sub_B46AA0(v30) )
      break;
    if ( j == v16 )
      return 0;
    LOWORD(v28) = 0;
  }
  v24 = (char *)v30;
  v9 = v87;
  v86 = v28;
  v4 = v84;
  if ( *v24 != 62
    || *((_QWORD *)v24 - 4) != *(_QWORD *)(v84 - 32)
    || !(unsigned __int8)sub_B50C50(
                           *(_QWORD *)(*((_QWORD *)v24 - 8) + 8LL),
                           *(_QWORD *)(*(_QWORD *)(v84 - 64) + 8LL),
                           *(_QWORD *)(a1 + 88))
    || !(unsigned __int8)sub_B45D20(v84, (__int64)v24, 0, 0, v31) )
  {
    return 0;
  }
LABEL_61:
  v81 = *((_QWORD *)v24 - 8);
  v32 = sub_B10CD0((__int64)(v24 + 48));
  v33 = sub_B10CD0(v4 + 48);
  v34 = sub_B026B0(v33, v32);
  sub_B10CB0(&v92, (__int64)v34);
  v35 = *(_QWORD *)(v4 - 64);
  if ( v35 != v81 )
  {
    v95[0] = "storemerge";
    v96 = 259;
    v77 = *(_QWORD *)(v35 + 8);
    v36 = sub_BD2DA0(80);
    v37 = v36;
    if ( v36 )
    {
      sub_B44260(v36, v77, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v37 + 72) = 2;
      sub_BD6B50((unsigned __int8 *)v37, v95);
      sub_BD2A10(v37, *(_DWORD *)(v37 + 72), 1);
    }
    v38 = *(_QWORD *)(v4 + 40);
    v39 = *(_QWORD *)(v4 - 64);
    v40 = *(_DWORD *)(v37 + 4) & 0x7FFFFFF;
    if ( v40 == *(_DWORD *)(v37 + 72) )
    {
      v76 = *(_QWORD *)(v4 + 40);
      v80 = *(_QWORD *)(v4 - 64);
      sub_B48D90(v37);
      v38 = v76;
      v39 = v80;
      v40 = *(_DWORD *)(v37 + 4) & 0x7FFFFFF;
    }
    v41 = (v40 + 1) & 0x7FFFFFF;
    v42 = v41 | *(_DWORD *)(v37 + 4) & 0xF8000000;
    v43 = *(_QWORD *)(v37 - 8) + 32LL * (unsigned int)(v41 - 1);
    *(_DWORD *)(v37 + 4) = v42;
    if ( *(_QWORD *)v43 )
    {
      v44 = *(_QWORD *)(v43 + 8);
      **(_QWORD **)(v43 + 16) = v44;
      if ( v44 )
        *(_QWORD *)(v44 + 16) = *(_QWORD *)(v43 + 16);
    }
    *(_QWORD *)v43 = v39;
    if ( v39 )
    {
      v45 = *(_QWORD *)(v39 + 16);
      *(_QWORD *)(v43 + 8) = v45;
      if ( v45 )
        *(_QWORD *)(v45 + 16) = v43 + 8;
      *(_QWORD *)(v43 + 16) = v39 + 16;
      *(_QWORD *)(v39 + 16) = v43;
    }
    *(_QWORD *)(*(_QWORD *)(v37 - 8)
              + 32LL * *(unsigned int *)(v37 + 72)
              + 8LL * ((*(_DWORD *)(v37 + 4) & 0x7FFFFFFu) - 1)) = v38;
    sub_D5F1F0(*(_QWORD *)(a1 + 32), (__int64)v24);
    v46 = *(__int64 **)(a1 + 32);
    v96 = 257;
    v47 = sub_10E0940(v46, v81, *(_QWORD *)(v37 + 8), (__int64)v95);
    v48 = *(_DWORD *)(v37 + 4) & 0x7FFFFFF;
    if ( v48 == *(_DWORD *)(v37 + 72) )
    {
      v82 = v47;
      sub_B48D90(v37);
      v47 = v82;
      v48 = *(_DWORD *)(v37 + 4) & 0x7FFFFFF;
    }
    v49 = (v48 + 1) & 0x7FFFFFF;
    v50 = v49 | *(_DWORD *)(v37 + 4) & 0xF8000000;
    v51 = *(_QWORD *)(v37 - 8) + 32LL * (unsigned int)(v49 - 1);
    *(_DWORD *)(v37 + 4) = v50;
    if ( *(_QWORD *)v51 )
    {
      v52 = *(_QWORD *)(v51 + 8);
      **(_QWORD **)(v51 + 16) = v52;
      if ( v52 )
        *(_QWORD *)(v52 + 16) = *(_QWORD *)(v51 + 16);
    }
    *(_QWORD *)v51 = v47;
    if ( v47 )
    {
      v53 = *(_QWORD *)(v47 + 16);
      *(_QWORD *)(v51 + 8) = v53;
      if ( v53 )
        *(_QWORD *)(v53 + 16) = v51 + 8;
      *(_QWORD *)(v51 + 16) = v47 + 16;
      *(_QWORD *)(v47 + 16) = v51;
    }
    *(_QWORD *)(*(_QWORD *)(v37 - 8)
              + 32LL * *(unsigned int *)(v37 + 72)
              + 8LL * ((*(_DWORD *)(v37 + 4) & 0x7FFFFFFu) - 1)) = v13;
    v54 = sub_114FCC0(a1, (_QWORD *)v37, *(_QWORD *)(v89 + 56), 1);
    v55 = (const char **)(v37 + 48);
    v81 = (__int64)v54;
    v95[0] = v92;
    if ( v92 )
    {
      sub_B96E90((__int64)v95, (__int64)v92, 1);
      v55 = (const char **)(v37 + 48);
      if ( (const char **)(v37 + 48) == v95 )
      {
        if ( v95[0] )
          sub_B91220((__int64)v95, (__int64)v95[0]);
        goto LABEL_86;
      }
      v71 = *(_QWORD *)(v37 + 48);
      if ( !v71 )
        goto LABEL_113;
    }
    else
    {
      if ( v55 == v95 )
        goto LABEL_86;
      v71 = *(_QWORD *)(v37 + 48);
      if ( !v71 )
        goto LABEL_86;
    }
    v79 = v55;
    sub_B91220((__int64)v55, v71);
    v55 = v79;
LABEL_113:
    v72 = (unsigned __int8 *)v95[0];
    *(const char **)(v37 + 48) = v95[0];
    if ( v72 )
      sub_B976B0((__int64)v95, v72, (__int64)v55);
  }
LABEL_86:
  v57 = sub_AA5190(v89);
  if ( v57 )
  {
    v58 = v56;
    v59 = HIBYTE(v56);
  }
  else
  {
    v59 = 0;
    v58 = 0;
  }
  LOBYTE(v60) = v58;
  BYTE1(v60) = v59;
  v61 = *(_WORD *)(v4 + 2);
  v90 = v86 | (unsigned __int16)v60;
  v73 = *(_QWORD *)(v4 - 32);
  v74 = v61 & 1;
  _BitScanReverse64(&v60, 1LL << (v61 >> 1));
  v78 = (v61 >> 7) & 7;
  v88 = *(_BYTE *)(v4 + 72);
  v75 = 63 - (v60 ^ 0x3F);
  v62 = sub_BD2C40(80, unk_3F10A10);
  v63 = v62;
  if ( v62 )
    sub_B4D260((__int64)v62, v81, v73, v74, v75, v78, v88, 0, 0);
  v64 = v57;
  v65 = (const char **)(v63 + 6);
  sub_114FCC0(a1, v63, v64, v90);
  v95[0] = v92;
  if ( v92 )
  {
    sub_B96E90((__int64)v95, (__int64)v92, 1);
    if ( v65 == v95 )
    {
      if ( v95[0] )
        sub_B91220((__int64)v95, (__int64)v95[0]);
      goto LABEL_94;
    }
    v69 = v63[6];
    if ( !v69 )
    {
LABEL_101:
      v70 = (unsigned __int8 *)v95[0];
      v63[6] = v95[0];
      if ( v70 )
        sub_B976B0((__int64)v95, v70, (__int64)(v63 + 6));
      goto LABEL_94;
    }
LABEL_100:
    sub_B91220((__int64)(v63 + 6), v69);
    goto LABEL_101;
  }
  if ( v65 != v95 )
  {
    v69 = v63[6];
    if ( v69 )
      goto LABEL_100;
  }
LABEL_94:
  v95[0] = (const char *)v4;
  v95[1] = v24;
  sub_AE9860((__int64)v63, (__int64)v95, 2);
  sub_B91FC0(v93, v4);
  if ( v93[0] || v93[1] || v93[2] || v93[3] )
  {
    sub_B91FC0(v94, (__int64)v24);
    sub_E01E30((__int64 *)v95, v93, v94, v66, v67, v68);
    sub_B9A100((__int64)v63, (__int64 *)v95);
  }
  sub_F207A0(a1, (__int64 *)v4);
  sub_F207A0(a1, (__int64 *)v24);
  if ( v92 )
    sub_B91220((__int64)&v92, (__int64)v92);
  return v9;
}
