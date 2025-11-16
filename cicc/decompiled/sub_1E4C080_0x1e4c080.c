// Function: sub_1E4C080
// Address: 0x1e4c080
//
void __fastcall sub_1E4C080(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9,
        unsigned int a10,
        char a11)
{
  __int64 v11; // r15
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  int v16; // ebx
  unsigned __int64 v17; // rsi
  _QWORD *v18; // rax
  _QWORD *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rdx
  unsigned int v22; // r12d
  int v23; // r8d
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned int v29; // ecx
  char v30; // si
  bool v31; // dl
  bool v32; // al
  __int64 v33; // rax
  unsigned int i; // ebx
  int v35; // r9d
  _DWORD *v36; // rax
  __int64 v37; // rax
  _WORD *v38; // rdx
  int v39; // esi
  __int64 v40; // r15
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // r15
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // r9
  int v47; // ecx
  __int64 v48; // rsi
  __int64 v49; // rax
  unsigned int v50; // esi
  __int64 v51; // r8
  unsigned int v52; // ecx
  __int64 *v53; // rdx
  __int64 v54; // rdi
  int v55; // edi
  __int64 v56; // r8
  __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // rax
  int v60; // eax
  int v61; // r12d
  __int64 *v62; // r10
  int v63; // edi
  __int64 v64; // rdi
  __int64 v65; // [rsp+10h] [rbp-110h]
  __int64 v66; // [rsp+18h] [rbp-108h]
  unsigned int v68; // [rsp+3Ch] [rbp-E4h]
  __int64 *v71; // [rsp+68h] [rbp-B8h]
  unsigned int v73; // [rsp+78h] [rbp-A8h]
  unsigned int v74; // [rsp+7Ch] [rbp-A4h]
  unsigned int v75; // [rsp+80h] [rbp-A0h]
  int v76; // [rsp+84h] [rbp-9Ch]
  __int64 v77; // [rsp+88h] [rbp-98h]
  __int32 v79; // [rsp+A0h] [rbp-80h]
  int v80; // [rsp+A4h] [rbp-7Ch]
  unsigned int v81; // [rsp+A4h] [rbp-7Ch]
  __int64 v82; // [rsp+A8h] [rbp-78h]
  unsigned int v83; // [rsp+B4h] [rbp-6Ch] BYREF
  __int64 v84; // [rsp+B8h] [rbp-68h] BYREF
  __m128i v85; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v86; // [rsp+D0h] [rbp-50h]
  __int64 v87; // [rsp+D8h] [rbp-48h]
  __int64 v88; // [rsp+E0h] [rbp-40h]

  v76 = a10 - a9;
  if ( a10 == a9 )
  {
    v73 = a9 - 1;
    v68 = a10;
  }
  else
  {
    v73 = 2 * a9 - a10;
    v68 = a10 - 1;
  }
  v11 = sub_1DD5D10(a1[115]);
  v66 = a1[115] + 24LL;
  if ( v11 == v66 )
    return;
  do
  {
    v13 = *(unsigned int *)(v11 + 40);
    if ( !(_DWORD)v13 )
      goto LABEL_59;
    v82 = v11;
    v14 = 0;
    v77 = 40 * v13;
    do
    {
      while ( 1 )
      {
        v15 = v14 + *(_QWORD *)(v82 + 32);
        if ( *(_BYTE *)v15 )
          goto LABEL_6;
        if ( (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
          goto LABEL_6;
        v16 = *(_DWORD *)(v15 + 8);
        if ( v16 >= 0 )
          goto LABEL_6;
        v17 = sub_1E45EB0((__int64)a1, v82);
        v18 = *(_QWORD **)(a6 + 48);
        if ( !v18 )
          goto LABEL_17;
        v19 = (_QWORD *)(a6 + 40);
        do
        {
          while ( 1 )
          {
            v20 = v18[2];
            v21 = v18[3];
            if ( v18[4] >= v17 )
              break;
            v18 = (_QWORD *)v18[3];
            if ( !v21 )
              goto LABEL_15;
          }
          v19 = v18;
          v18 = (_QWORD *)v18[2];
        }
        while ( v20 );
LABEL_15:
        if ( (_QWORD *)(a6 + 40) != v19 && v19[4] <= v17 )
        {
          v23 = (*((_DWORD *)v19 + 10) - *(_DWORD *)(a6 + 128)) / *(_DWORD *)(a6 + 136);
          v22 = v23;
        }
        else
        {
LABEL_17:
          v22 = -1;
          v23 = -1;
        }
        v83 = v16;
        LODWORD(v84) = v16;
        v24 = a6 + 88;
        v25 = *(_QWORD *)(a6 + 96);
        if ( !v25 )
          goto LABEL_25;
        do
        {
          while ( 1 )
          {
            v26 = *(_QWORD *)(v25 + 16);
            v27 = *(_QWORD *)(v25 + 24);
            if ( (unsigned int)v16 <= *(_DWORD *)(v25 + 32) )
              break;
            v25 = *(_QWORD *)(v25 + 24);
            if ( !v27 )
              goto LABEL_23;
          }
          v24 = v25;
          v25 = *(_QWORD *)(v25 + 16);
        }
        while ( v26 );
LABEL_23:
        if ( a6 + 88 == v24 || (unsigned int)v16 < *(_DWORD *)(v24 + 32) )
        {
LABEL_25:
          v80 = v23;
          v85.m128i_i64[0] = (__int64)&v84;
          v28 = sub_1E48710((_QWORD *)(a6 + 80), v24, (unsigned int **)&v85);
          v23 = v80;
          v24 = v28;
        }
        v29 = *(_DWORD *)(v24 + 36);
        v30 = *(_BYTE *)(v24 + 40);
        if ( a10 > (*(_DWORD *)(a6 + 132) - *(_DWORD *)(a6 + 128)) / *(_DWORD *)(a6 + 136) )
        {
          if ( v29 )
            goto LABEL_84;
          if ( v30 )
          {
            v29 = 1;
LABEL_84:
            v32 = a10 != a9;
            goto LABEL_29;
          }
          v31 = a10 != a9;
          v32 = a10 != a9;
          if ( !v76 )
          {
LABEL_79:
            v32 = v31;
            goto LABEL_29;
          }
        }
        else
        {
          v31 = a10 != a9;
          v32 = a10 != a9 && v29 == 0;
          if ( !v32 )
            goto LABEL_79;
        }
        v29 = 0;
        if ( !v23 )
          break;
LABEL_29:
        if ( v73 >= v22 || !v32 )
        {
          if ( v73 + 1 - v22 <= v29 )
            v29 = v73 + 1 - v22;
          v74 = v29;
          goto LABEL_34;
        }
LABEL_6:
        v14 += 40;
        if ( v77 == v14 )
          goto LABEL_58;
      }
      v58 = a1[5];
      if ( (v83 & 0x80000000) != 0 )
        v59 = *(_QWORD *)(*(_QWORD *)(v58 + 24) + 16LL * (v83 & 0x7FFFFFFF) + 8);
      else
        v59 = *(_QWORD *)(*(_QWORD *)(v58 + 272) + 8LL * v83);
      v74 = 0;
      if ( v59 )
      {
        if ( (*(_BYTE *)(v59 + 3) & 0x10) != 0 )
        {
          v59 = *(_QWORD *)(v59 + 32);
          if ( !v59 )
            goto LABEL_34;
          while ( (*(_BYTE *)(v59 + 3) & 0x10) != 0 )
          {
            v59 = *(_QWORD *)(v59 + 32);
            if ( !v59 )
            {
              v74 = 0;
              goto LABEL_34;
            }
          }
        }
LABEL_97:
        if ( a1[115] == *(_QWORD *)(*(_QWORD *)(v59 + 16) + 24LL) )
        {
          while ( 1 )
          {
            v59 = *(_QWORD *)(v59 + 32);
            if ( !v59 )
              break;
            if ( (*(_BYTE *)(v59 + 3) & 0x10) == 0 )
              goto LABEL_97;
          }
          v74 = 0;
        }
        else
        {
          v60 = 1;
          if ( v73 + 1 == v22 )
            v60 = v73 + 1 - v22;
          v74 = v60;
        }
      }
LABEL_34:
      v75 = sub_1E49390(a7 + 32LL * v68, (int *)&v83)[1];
      v33 = sub_1E69D00(a1[5], v75);
      if ( v33 && (**(_WORD **)(v33 + 16) == 45 || !**(_WORD **)(v33 + 16)) && a2 == *(_QWORD *)(v33 + 24) )
        v75 = sub_1E40FE0(*(_QWORD *)(v33 + 32), *(_DWORD *)(v33 + 40), a4);
      if ( !v74 )
        goto LABEL_6;
      v65 = v14;
      for ( i = 0; i != v74; ++i )
      {
        while ( 1 )
        {
          v36 = sub_1E49390(a7 + 32LL * v73, (int *)&v83);
          if ( v73 >= i )
            v36 = sub_1E49390(a7 + 32LL * (v73 - i), (int *)&v83);
          v81 = v36[1];
          v37 = sub_1E69D00(a1[5], v81);
          if ( !v37 )
            goto LABEL_49;
          v38 = *(_WORD **)(v37 + 16);
          v39 = (unsigned __int16)*v38;
          if ( *v38 )
          {
            if ( v39 != 45 )
              goto LABEL_49;
          }
          v46 = *(_QWORD *)(v37 + 24);
          if ( a5 == v46 )
          {
            v55 = *(_DWORD *)(v37 + 40);
            v56 = *(_QWORD *)(v37 + 32);
            v57 = 1;
            if ( v55 == 1 )
            {
              v81 = 0;
            }
            else
            {
              do
              {
                if ( a5 != *(_QWORD *)(v56 + 40LL * (unsigned int)(v57 + 1) + 24) )
                {
                  v81 = *(_DWORD *)(v56 + 40 * v57 + 8);
                  goto LABEL_90;
                }
                v57 = (unsigned int)(v57 + 2);
              }
              while ( v55 != (_DWORD)v57 );
              v81 = 0;
LABEL_90:
              if ( v39 && v39 != 45 )
                goto LABEL_49;
            }
          }
          if ( a2 != v46 )
            goto LABEL_49;
          v47 = *(_DWORD *)(v37 + 40);
          v48 = *(_QWORD *)(v37 + 32);
          if ( v47 == 1 )
          {
LABEL_102:
            v81 = 0;
LABEL_49:
            if ( !v76 )
              goto LABEL_50;
LABEL_70:
            v75 = sub_1E49390(a7 + 32LL * (v68 - i), (int *)&v83)[1];
            goto LABEL_50;
          }
          v49 = 1;
          while ( a2 == *(_QWORD *)(v48 + 40LL * (unsigned int)(v49 + 1) + 24) )
          {
            v49 = (unsigned int)(v49 + 2);
            if ( v47 == (_DWORD)v49 )
              goto LABEL_102;
          }
          v81 = *(_DWORD *)(v48 + 40 * v49 + 8);
          if ( v76 )
            goto LABEL_70;
LABEL_50:
          v79 = sub_1E6B9A0(
                  a1[5],
                  *(_QWORD *)(*(_QWORD *)(a1[5] + 24LL) + 16LL * (v83 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                  byte_3F871B3,
                  0);
          v40 = *(_QWORD *)(a1[2] + 8LL);
          v84 = 0;
          v41 = sub_1DD5D10(a2);
          v42 = *(_QWORD *)(a2 + 56);
          v71 = (__int64 *)v41;
          v43 = (__int64)sub_1E0B640(v42, v40, &v84, 0);
          sub_1DD5BA0((__int64 *)(a2 + 16), v43);
          v44 = *v71;
          v45 = *(_QWORD *)v43 & 7LL;
          *(_QWORD *)(v43 + 8) = v71;
          v44 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v43 = v44 | v45;
          *(_QWORD *)(v44 + 8) = v43;
          *v71 = v43 | *v71 & 7;
          v85.m128i_i64[0] = 0x10000000;
          v86 = 0;
          v85.m128i_i32[2] = v79;
          v87 = 0;
          v88 = 0;
          sub_1E1A9C0(v43, v42, &v85);
          if ( v84 )
            sub_161E7C0((__int64)&v84, v84);
          v85.m128i_i64[0] = 0;
          v85.m128i_i32[2] = v81;
          v86 = 0;
          v87 = 0;
          v88 = 0;
          sub_1E1A9C0(v43, v42, &v85);
          v85.m128i_i8[0] = 4;
          v86 = 0;
          v85.m128i_i32[0] &= 0xFFF000FF;
          v87 = a3;
          sub_1E1A9C0(v43, v42, &v85);
          v85.m128i_i64[0] = 0;
          v86 = 0;
          v85.m128i_i32[2] = v75;
          v87 = 0;
          v88 = 0;
          sub_1E1A9C0(v43, v42, &v85);
          v85.m128i_i8[0] = 4;
          v86 = 0;
          v85.m128i_i32[0] &= 0xFFF000FF;
          v87 = a4;
          sub_1E1A9C0(v43, v42, &v85);
          if ( !i )
          {
            v84 = v43;
            v50 = *(_DWORD *)(a8 + 24);
            if ( v50 )
            {
              v51 = *(_QWORD *)(a8 + 8);
              v52 = (v50 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
              v53 = (__int64 *)(v51 + 16LL * v52);
              v54 = *v53;
              if ( v43 == *v53 )
              {
LABEL_82:
                v53[1] = v82;
                goto LABEL_53;
              }
              v61 = 1;
              v62 = 0;
              while ( v54 != -8 )
              {
                if ( !v62 && v54 == -16 )
                  v62 = v53;
                v52 = (v50 - 1) & (v61 + v52);
                v53 = (__int64 *)(v51 + 16LL * v52);
                v54 = *v53;
                if ( v43 == *v53 )
                  goto LABEL_82;
                ++v61;
              }
              if ( v62 )
                v53 = v62;
              ++*(_QWORD *)a8;
              v63 = *(_DWORD *)(a8 + 16) + 1;
              if ( 4 * v63 < 3 * v50 )
              {
                if ( v50 - *(_DWORD *)(a8 + 20) - v63 > v50 >> 3 )
                {
LABEL_119:
                  *(_DWORD *)(a8 + 16) = v63;
                  if ( *v53 != -8 )
                    --*(_DWORD *)(a8 + 20);
                  *v53 = v43;
                  v53[1] = 0;
                  goto LABEL_82;
                }
                v64 = a8;
LABEL_124:
                sub_1E4BEC0(v64, v50);
                sub_1E48B10(a8, &v84, &v85);
                v53 = (__int64 *)v85.m128i_i64[0];
                v43 = v84;
                v63 = *(_DWORD *)(a8 + 16) + 1;
                goto LABEL_119;
              }
            }
            else
            {
              ++*(_QWORD *)a8;
            }
            v64 = a8;
            v50 *= 2;
            goto LABEL_124;
          }
LABEL_53:
          if ( v76 )
            break;
          sub_1E46110((__int64)a1, a2, a6, a8, a10, i, v82, v81, v79, 0);
          sub_1E46110((__int64)a1, a2, a6, a8, a10, i, v82, v75, v79, 0);
          sub_1E49390(a7 + 32LL * (v68 - 1 - i), (int *)&v83)[1] = v79;
          if ( a11 )
          {
            v75 = v79;
            if ( v74 - 1 == i )
              goto LABEL_56;
          }
          else
          {
            v75 = v79;
          }
LABEL_43:
          if ( ++i == v74 )
            goto LABEL_57;
        }
        sub_1E49390(a7 + 32LL * (a10 - i), (int *)&v83)[1] = v79;
        if ( v74 - 1 != i )
          goto LABEL_43;
        sub_1E46110((__int64)a1, a2, a6, a8, a10, i, v82, v83, v79, 0);
        if ( !a11 )
          goto LABEL_43;
LABEL_56:
        sub_1E42770(v83, v79, a1[115], a1[5], a1[266], v35);
      }
LABEL_57:
      v14 = v65 + 40;
    }
    while ( v77 != v65 + 40 );
LABEL_58:
    v11 = v82;
LABEL_59:
    if ( (*(_BYTE *)v11 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v11 + 46) & 8) != 0 )
        v11 = *(_QWORD *)(v11 + 8);
    }
    v11 = *(_QWORD *)(v11 + 8);
  }
  while ( v66 != v11 );
}
