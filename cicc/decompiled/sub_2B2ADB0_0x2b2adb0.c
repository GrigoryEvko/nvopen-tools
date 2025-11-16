// Function: sub_2B2ADB0
// Address: 0x2b2adb0
//
__int64 __fastcall sub_2B2ADB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdx
  int v14; // esi
  int v16; // esi
  __int64 v17; // rax
  __int64 v18; // rsi
  _BYTE **v19; // r13
  __int64 v20; // rax
  __int64 v21; // rsi
  _BYTE **v22; // r11
  _BYTE *v23; // rax
  char v24; // si
  unsigned int v25; // ecx
  __int64 v26; // rdx
  _BYTE *v27; // r10
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 i; // rdx
  _DWORD *v31; // rcx
  _DWORD *v32; // r13
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdx
  int v35; // ecx
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // rax
  int v39; // esi
  int v40; // esi
  _BYTE *v41; // rax
  char v42; // si
  unsigned int v43; // ecx
  __int64 v44; // rdx
  _BYTE *v45; // r10
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  _BYTE *v49; // rax
  char v50; // si
  unsigned int v51; // ecx
  __int64 v52; // rdx
  _BYTE *v53; // r10
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdx
  _BYTE *v57; // rax
  char v58; // si
  unsigned int v59; // ecx
  __int64 v60; // rdx
  _BYTE *v61; // r10
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rcx
  _DWORD *v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // rdx
  __int64 v70; // rdx
  __int64 v71; // rdx
  int v72; // edx
  int v73; // edx
  int v74; // edx
  int v75; // edx
  int v76; // edx
  __int64 v77; // rax
  _QWORD *v78; // r12
  int v79; // edx
  unsigned __int64 v80; // r12
  unsigned __int64 *v81; // rax
  int v82; // edx
  unsigned __int64 v83; // [rsp+0h] [rbp-C0h]
  int v84; // [rsp+8h] [rbp-B8h]
  int v85; // [rsp+8h] [rbp-B8h]
  int v86; // [rsp+8h] [rbp-B8h]
  int v87; // [rsp+8h] [rbp-B8h]
  __int64 v88; // [rsp+10h] [rbp-B0h]
  __int64 v90; // [rsp+20h] [rbp-A0h]
  _BYTE **v91; // [rsp+38h] [rbp-88h]
  __int64 v92; // [rsp+40h] [rbp-80h]
  int v94; // [rsp+5Ch] [rbp-64h] BYREF
  __int64 v95; // [rsp+60h] [rbp-60h] BYREF
  _DWORD *v96; // [rsp+68h] [rbp-58h] BYREF
  __int64 v97[10]; // [rsp+70h] [rbp-50h] BYREF

  v90 = a6;
  v92 = *(unsigned int *)(a2 + 248);
  if ( !(_DWORD)v92 )
    return 1;
  v7 = 0;
  v88 = a4 + 8 * a5;
  while ( 1 )
  {
    v8 = *(_QWORD *)a3;
    v9 = 16LL * *(unsigned int *)(a3 + 8);
    v10 = *(_QWORD *)a3 + v9;
    v11 = v9 >> 4;
    v12 = v9 >> 6;
    if ( v12 )
    {
      v13 = v8 + (v12 << 6);
      while ( 1 )
      {
        if ( *(_DWORD *)v8 == (_DWORD)v7 )
        {
          v14 = *(_DWORD *)(*(_QWORD *)(v8 + 8) + 104LL);
          if ( (v14 & 0xFFFFFFFD) == 0 || v14 == 5 )
            goto LABEL_12;
        }
        if ( *(_DWORD *)(v8 + 16) == (_DWORD)v7 )
        {
          v16 = *(_DWORD *)(*(_QWORD *)(v8 + 24) + 104LL);
          if ( (v16 & 0xFFFFFFFD) == 0 || v16 == 5 )
          {
            if ( v10 != v8 + 16 )
              goto LABEL_13;
            goto LABEL_18;
          }
        }
        if ( *(_DWORD *)(v8 + 32) == (_DWORD)v7 )
        {
          v39 = *(_DWORD *)(*(_QWORD *)(v8 + 40) + 104LL);
          if ( (v39 & 0xFFFFFFFD) == 0 || v39 == 5 )
          {
            v8 += 32;
            goto LABEL_12;
          }
        }
        if ( *(_DWORD *)(v8 + 48) == (_DWORD)v7 )
        {
          v40 = *(_DWORD *)(*(_QWORD *)(v8 + 56) + 104LL);
          if ( (v40 & 0xFFFFFFFD) == 0 || v40 == 5 )
          {
            v8 += 48;
            goto LABEL_12;
          }
        }
        v8 += 64;
        if ( v13 == v8 )
        {
          v11 = (v10 - v8) >> 4;
          break;
        }
      }
    }
    if ( v11 == 2 )
      goto LABEL_153;
    if ( v11 != 3 )
    {
      if ( v11 != 1 || *(_DWORD *)v8 != (_DWORD)v7 )
        goto LABEL_18;
      goto LABEL_130;
    }
    if ( *(_DWORD *)v8 != (_DWORD)v7
      || (v82 = *(_DWORD *)(*(_QWORD *)(v8 + 8) + 104LL), (v82 & 0xFFFFFFFD) != 0) && v82 != 5 )
    {
      v8 += 16;
LABEL_153:
      if ( *(_DWORD *)v8 != (_DWORD)v7
        || (v79 = *(_DWORD *)(*(_QWORD *)(v8 + 8) + 104LL), (v79 & 0xFFFFFFFD) != 0) && v79 != 5 )
      {
        v8 += 16;
        if ( *(_DWORD *)v8 != (_DWORD)v7 )
          goto LABEL_18;
LABEL_130:
        v72 = *(_DWORD *)(*(_QWORD *)(v8 + 8) + 104LL);
        if ( (v72 & 0xFFFFFFFD) != 0 && v72 != 5 )
          goto LABEL_18;
      }
    }
LABEL_12:
    if ( v10 != v8 )
      goto LABEL_13;
LABEL_18:
    v95 = a2;
    v94 = v7;
    v17 = *(_QWORD *)(a2 + 240) + 80 * v7;
    v18 = *(unsigned int *)(v17 + 8);
    v19 = *(_BYTE ***)v17;
    v96 = 0;
    v97[0] = a1;
    v18 *= 8;
    v91 = (_BYTE **)((char *)v19 + v18);
    v97[1] = (__int64)&v95;
    v97[2] = (__int64)&v94;
    v97[3] = (__int64)&v96;
    v20 = v18 >> 3;
    v21 = v18 >> 5;
    if ( !v21 )
      break;
    v22 = &v19[4 * v21];
    while ( 1 )
    {
      v23 = *v19;
      if ( **v19 <= 0x1Cu )
        goto LABEL_52;
      v24 = *(_BYTE *)(a1 + 88) & 1;
      if ( v24 )
      {
        a6 = a1 + 96;
        a5 = 3;
      }
      else
      {
        a5 = *(unsigned int *)(a1 + 104);
        a6 = *(_QWORD *)(a1 + 96);
        if ( !(_DWORD)a5 )
          goto LABEL_114;
        a5 = (unsigned int)(a5 - 1);
      }
      v25 = a5 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v26 = a6 + 72LL * v25;
      v27 = *(_BYTE **)v26;
      if ( *(_BYTE **)v26 == v23 )
        goto LABEL_24;
      v73 = 1;
      while ( v27 != (_BYTE *)-4096LL )
      {
        v25 = a5 & (v73 + v25);
        v84 = v73 + 1;
        v26 = a6 + 72LL * v25;
        v27 = *(_BYTE **)v26;
        if ( v23 == *(_BYTE **)v26 )
          goto LABEL_24;
        v73 = v84;
      }
      if ( v24 )
      {
        v68 = 288;
        goto LABEL_115;
      }
      a5 = *(unsigned int *)(a1 + 104);
LABEL_114:
      v68 = 72 * a5;
LABEL_115:
      v26 = a6 + v68;
LABEL_24:
      v28 = 288;
      if ( !v24 )
        v28 = 72LL * *(unsigned int *)(a1 + 104);
      if ( v26 != a6 + v28 )
      {
        v29 = *(_QWORD *)(v26 + 8);
        for ( i = v29 + 8LL * *(unsigned int *)(v26 + 16); i != v29; v29 += 8 )
        {
          v31 = *(_DWORD **)v29;
          if ( a2 == *(_QWORD *)(*(_QWORD *)v29 + 184LL) && v31[48] == (_DWORD)v7 )
            goto LABEL_32;
        }
      }
LABEL_52:
      v41 = v19[1];
      if ( *v41 <= 0x1Cu )
        goto LABEL_65;
      v42 = *(_BYTE *)(a1 + 88) & 1;
      if ( v42 )
      {
        a6 = a1 + 96;
        a5 = 3;
      }
      else
      {
        a5 = *(unsigned int *)(a1 + 104);
        a6 = *(_QWORD *)(a1 + 96);
        if ( !(_DWORD)a5 )
          goto LABEL_117;
        a5 = (unsigned int)(a5 - 1);
      }
      v43 = a5 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
      v44 = a6 + 72LL * v43;
      v45 = *(_BYTE **)v44;
      if ( v41 == *(_BYTE **)v44 )
        goto LABEL_56;
      v74 = 1;
      while ( v45 != (_BYTE *)-4096LL )
      {
        v43 = a5 & (v74 + v43);
        v85 = v74 + 1;
        v44 = a6 + 72LL * v43;
        v45 = *(_BYTE **)v44;
        if ( v41 == *(_BYTE **)v44 )
          goto LABEL_56;
        v74 = v85;
      }
      if ( v42 )
      {
        v69 = 288;
        goto LABEL_118;
      }
      a5 = *(unsigned int *)(a1 + 104);
LABEL_117:
      v69 = 72 * a5;
LABEL_118:
      v44 = a6 + v69;
LABEL_56:
      v46 = 288;
      if ( !v42 )
        v46 = 72LL * *(unsigned int *)(a1 + 104);
      if ( v44 != a6 + v46 )
      {
        v47 = *(_QWORD *)(v44 + 8);
        v48 = v47 + 8LL * *(unsigned int *)(v44 + 16);
        if ( v48 != v47 )
        {
          while ( 1 )
          {
            v31 = *(_DWORD **)v47;
            if ( a2 == *(_QWORD *)(*(_QWORD *)v47 + 184LL) && v31[48] == (_DWORD)v7 )
              break;
            v47 += 8;
            if ( v48 == v47 )
              goto LABEL_65;
          }
          ++v19;
LABEL_32:
          v96 = v31;
          goto LABEL_33;
        }
      }
LABEL_65:
      v49 = v19[2];
      if ( *v49 <= 0x1Cu )
        goto LABEL_78;
      v50 = *(_BYTE *)(a1 + 88) & 1;
      if ( v50 )
      {
        a6 = a1 + 96;
        a5 = 3;
      }
      else
      {
        a5 = *(unsigned int *)(a1 + 104);
        a6 = *(_QWORD *)(a1 + 96);
        if ( !(_DWORD)a5 )
          goto LABEL_120;
        a5 = (unsigned int)(a5 - 1);
      }
      v51 = a5 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
      v52 = a6 + 72LL * v51;
      v53 = *(_BYTE **)v52;
      if ( *(_BYTE **)v52 == v49 )
        goto LABEL_69;
      v75 = 1;
      while ( v53 != (_BYTE *)-4096LL )
      {
        v51 = a5 & (v75 + v51);
        v87 = v75 + 1;
        v52 = a6 + 72LL * v51;
        v53 = *(_BYTE **)v52;
        if ( v49 == *(_BYTE **)v52 )
          goto LABEL_69;
        v75 = v87;
      }
      if ( v50 )
      {
        v70 = 288;
        goto LABEL_121;
      }
      a5 = *(unsigned int *)(a1 + 104);
LABEL_120:
      v70 = 72 * a5;
LABEL_121:
      v52 = a6 + v70;
LABEL_69:
      v54 = 288;
      if ( !v50 )
        v54 = 72LL * *(unsigned int *)(a1 + 104);
      if ( v52 != a6 + v54 )
      {
        v55 = *(_QWORD *)(v52 + 8);
        v56 = v55 + 8LL * *(unsigned int *)(v52 + 16);
        if ( v56 != v55 )
          break;
      }
LABEL_78:
      v57 = v19[3];
      if ( *v57 <= 0x1Cu )
        goto LABEL_91;
      v58 = *(_BYTE *)(a1 + 88) & 1;
      if ( v58 )
      {
        a6 = a1 + 96;
        a5 = 3;
      }
      else
      {
        a5 = *(unsigned int *)(a1 + 104);
        a6 = *(_QWORD *)(a1 + 96);
        if ( !(_DWORD)a5 )
          goto LABEL_123;
        a5 = (unsigned int)(a5 - 1);
      }
      v59 = a5 & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
      v60 = a6 + 72LL * v59;
      v61 = *(_BYTE **)v60;
      if ( v57 == *(_BYTE **)v60 )
        goto LABEL_82;
      v76 = 1;
      while ( v61 != (_BYTE *)-4096LL )
      {
        v59 = a5 & (v76 + v59);
        v86 = v76 + 1;
        v60 = a6 + 72LL * v59;
        v61 = *(_BYTE **)v60;
        if ( v57 == *(_BYTE **)v60 )
          goto LABEL_82;
        v76 = v86;
      }
      if ( v58 )
      {
        v71 = 288;
        goto LABEL_124;
      }
      a5 = *(unsigned int *)(a1 + 104);
LABEL_123:
      v71 = 72 * a5;
LABEL_124:
      v60 = a6 + v71;
LABEL_82:
      v62 = 288;
      if ( !v58 )
        v62 = 72LL * *(unsigned int *)(a1 + 104);
      if ( v60 != a6 + v62 )
      {
        v63 = *(_QWORD *)(v60 + 8);
        v64 = v63 + 8LL * *(unsigned int *)(v60 + 16);
        if ( v64 != v63 )
        {
          while ( a2 != *(_QWORD *)(*(_QWORD *)v63 + 184LL) || *(_DWORD *)(*(_QWORD *)v63 + 192LL) != (_DWORD)v7 )
          {
            v63 += 8;
            if ( v64 == v63 )
              goto LABEL_91;
          }
          v96 = *(_DWORD **)v63;
          v19 += 3;
          goto LABEL_33;
        }
      }
LABEL_91:
      v19 += 4;
      if ( v22 == v19 )
      {
        v20 = v91 - v19;
        goto LABEL_93;
      }
    }
    while ( a2 != *(_QWORD *)(*(_QWORD *)v55 + 184LL) || *(_DWORD *)(*(_QWORD *)v55 + 192LL) != (_DWORD)v7 )
    {
      v55 += 8;
      if ( v56 == v55 )
        goto LABEL_78;
    }
    v96 = *(_DWORD **)v55;
    v19 += 2;
LABEL_33:
    if ( v91 == v19 )
      goto LABEL_96;
    v32 = v96;
    if ( !v96 )
      goto LABEL_96;
    v33 = *(unsigned int *)(a3 + 8);
    v34 = *(unsigned int *)(a3 + 12);
    v35 = *(_DWORD *)(a3 + 8);
    if ( v33 >= v34 )
    {
      a5 = v33 + 1;
      v80 = v83 & 0xFFFFFFFF00000000LL | (unsigned int)v7;
      v83 = v80;
      if ( v34 < v33 + 1 )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v33 + 1, 0x10u, a5, a6);
        v33 = *(unsigned int *)(a3 + 8);
      }
      v81 = (unsigned __int64 *)(*(_QWORD *)a3 + 16 * v33);
      *v81 = v80;
      v81[1] = (unsigned __int64)v32;
      ++*(_DWORD *)(a3 + 8);
    }
    else
    {
      v36 = *(_QWORD *)a3 + 16 * v33;
      if ( v36 )
      {
        *(_DWORD *)v36 = v7;
        *(_QWORD *)(v36 + 8) = v32;
        v35 = *(_DWORD *)(a3 + 8);
      }
      *(_DWORD *)(a3 + 8) = v35 + 1;
    }
    v37 = v32[26];
    if ( (v37 & 0xFFFFFFFD) != 0 && v37 != 5 && !v32[30] && !v32[38] )
    {
LABEL_43:
      v38 = *(unsigned int *)(v90 + 8);
      if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(v90 + 12) )
      {
        sub_C8D5F0(v90, (const void *)(v90 + 16), v38 + 1, 8u, a5, a6);
        v38 = *(unsigned int *)(v90 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v90 + 8 * v38) = v32;
      ++*(_DWORD *)(v90 + 8);
    }
LABEL_13:
    if ( v92 == ++v7 )
      return 1;
  }
LABEL_93:
  switch ( v20 )
  {
    case 2LL:
LABEL_158:
      if ( (unsigned __int8)sub_2B29CF0(v97, *v19) )
        goto LABEL_33;
      ++v19;
      goto LABEL_160;
    case 3LL:
      if ( (unsigned __int8)sub_2B29CF0(v97, *v19) )
        goto LABEL_33;
      ++v19;
      goto LABEL_158;
    case 1LL:
LABEL_160:
      if ( (unsigned __int8)sub_2B29CF0(v97, *v19) )
        goto LABEL_33;
      break;
  }
LABEL_96:
  v65 = a4;
  if ( a4 == v88 )
    goto LABEL_13;
  v32 = 0;
  v66 = 0;
  do
  {
    while ( 1 )
    {
      v67 = *(_DWORD **)v65;
      if ( a2 == *(_QWORD *)(*(_QWORD *)v65 + 184LL) && v67[48] == (_DWORD)v7 )
        break;
      v65 += 8;
      if ( v88 == v65 )
        goto LABEL_102;
    }
    v65 += 8;
    ++v66;
    v32 = v67;
  }
  while ( v88 != v65 );
LABEL_102:
  if ( v66 <= 1
    || (v77 = *(_QWORD *)(a2 + 240) + 80 * v7,
        v78 = (_QWORD *)(*(_QWORD *)v77 + 8LL * *(unsigned int *)(v77 + 8)),
        v78 == sub_2B0BF30(*(_QWORD **)v77, (__int64)v78, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0)) )
  {
    if ( v32 )
      goto LABEL_43;
    goto LABEL_13;
  }
  return 0;
}
