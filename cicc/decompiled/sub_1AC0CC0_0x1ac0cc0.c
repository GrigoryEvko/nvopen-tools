// Function: sub_1AC0CC0
// Address: 0x1ac0cc0
//
__int64 __fastcall sub_1AC0CC0(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  __int64 v12; // r12
  __int64 result; // rax
  __int64 v14; // rsi
  int v15; // r10d
  unsigned int v16; // r11d
  __int64 v17; // r15
  __int64 v18; // r8
  int v19; // r12d
  __int64 v20; // rcx
  __int64 v21; // r13
  __int64 v22; // rdx
  int v23; // edi
  __int64 v24; // rcx
  __int64 v25; // r14
  __int64 v26; // r13
  __int64 v27; // r12
  __int64 v28; // rax
  int v29; // eax
  __int64 v30; // rdi
  int v31; // edx
  unsigned int v32; // eax
  __int64 *v33; // rcx
  __int64 v34; // rsi
  char *v35; // rdx
  char *v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rcx
  char *v39; // rax
  __int64 v40; // r12
  __int64 v41; // rbx
  __int64 v42; // r13
  __int64 v43; // r15
  __int64 v44; // r12
  __int64 v45; // rsi
  int v46; // eax
  __int64 v47; // r8
  int v48; // r9d
  __int64 v49; // rdi
  int v50; // edx
  unsigned int v51; // eax
  __int64 v52; // rsi
  unsigned __int64 v53; // rax
  __int64 v54; // r9
  __int64 v55; // rdx
  __int64 v56; // rax
  double v57; // xmm4_8
  double v58; // xmm5_8
  __int64 v59; // r9
  __int64 v60; // rbx
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  int v65; // eax
  __int64 v66; // rax
  int v67; // edx
  __int64 v68; // rdx
  _QWORD *v69; // rax
  __int64 v70; // rcx
  unsigned __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r9
  __int64 v76; // r13
  __int64 v77; // r8
  unsigned int v78; // ebx
  __int64 v79; // rcx
  int v80; // edx
  __int64 v81; // rdx
  __int64 v82; // r10
  int v83; // r11d
  __int64 v84; // r15
  unsigned int v85; // eax
  __int64 v86; // rdi
  __int64 v87; // rcx
  int v88; // eax
  __int64 v89; // rax
  int v90; // edx
  __int64 v91; // rdx
  __int64 *v92; // rax
  __int64 v93; // rsi
  unsigned __int64 v94; // rdx
  __int64 v95; // rdx
  __int64 v96; // rdx
  __int64 v97; // rcx
  __int64 v98; // r13
  __int64 v99; // rax
  int v100; // ecx
  int v101; // r8d
  __int64 v102; // [rsp+8h] [rbp-98h]
  int v103; // [rsp+10h] [rbp-90h]
  __int64 v104; // [rsp+18h] [rbp-88h]
  int v105; // [rsp+18h] [rbp-88h]
  __int64 v106; // [rsp+20h] [rbp-80h]
  __int64 v107; // [rsp+20h] [rbp-80h]
  __int64 v108; // [rsp+20h] [rbp-80h]
  __int64 v110; // [rsp+28h] [rbp-78h]
  __int64 v111; // [rsp+28h] [rbp-78h]
  __int64 v112; // [rsp+28h] [rbp-78h]
  __int64 v113; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v114[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v115[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v116; // [rsp+60h] [rbp-40h]

  v10 = a1;
  v12 = *a2;
  result = *(_QWORD *)(*(_QWORD *)(*a2 + 56) + 80LL);
  if ( result )
  {
    result -= 24;
    if ( v12 == result )
    {
      v98 = *(_QWORD *)a1;
      v27 = a1 + 40;
      v99 = sub_157ED20(*a2);
      v113 = sub_1AA8CA0((_QWORD *)*a2, v99, v98, 0);
      v29 = *(_DWORD *)(a1 + 64);
      if ( !v29 )
      {
        sub_1ABFB80(a1 + 40, &v113);
        result = v113;
        *a2 = v113;
        return result;
      }
      v30 = *(_QWORD *)(a1 + 48);
      v103 = 0;
      v104 = *a2;
LABEL_16:
      v31 = v29 - 1;
      v32 = (v29 - 1) & (((unsigned int)v104 >> 9) ^ ((unsigned int)v104 >> 4));
      v33 = (__int64 *)(v30 + 8LL * v32);
      v34 = *v33;
      if ( v104 != *v33 )
      {
        v100 = 1;
        while ( v34 != -8 )
        {
          v101 = v100 + 1;
          v32 = v31 & (v100 + v32);
          v33 = (__int64 *)(v30 + 8LL * v32);
          v34 = *v33;
          if ( *v33 == v104 )
            goto LABEL_17;
          v100 = v101;
        }
        goto LABEL_27;
      }
LABEL_17:
      *v33 = -16;
      v35 = *(char **)(v10 + 80);
      v36 = *(char **)(v10 + 72);
      --*(_DWORD *)(v10 + 56);
      ++*(_DWORD *)(v10 + 60);
      v37 = (v35 - v36) >> 5;
      v38 = (v35 - v36) >> 3;
      if ( v37 > 0 )
      {
        v39 = &v36[32 * v37];
        while ( v104 != *(_QWORD *)v36 )
        {
          if ( v104 == *((_QWORD *)v36 + 1) )
          {
            v36 += 8;
            goto LABEL_24;
          }
          if ( v104 == *((_QWORD *)v36 + 2) )
          {
            v36 += 16;
            goto LABEL_24;
          }
          if ( v104 == *((_QWORD *)v36 + 3) )
          {
            v36 += 24;
            goto LABEL_24;
          }
          v36 += 32;
          if ( v36 == v39 )
          {
            v38 = (v35 - v36) >> 3;
            goto LABEL_104;
          }
        }
        goto LABEL_24;
      }
LABEL_104:
      if ( v38 != 2 )
      {
        if ( v38 != 3 )
        {
          if ( v38 != 1 )
          {
            v36 = v35;
LABEL_24:
            if ( v36 + 8 != v35 )
            {
              memmove(v36, v36 + 8, v35 - (v36 + 8));
              v35 = *(char **)(v10 + 80);
            }
            *(_QWORD *)(v10 + 80) = v35 - 8;
            goto LABEL_27;
          }
LABEL_115:
          if ( v104 != *(_QWORD *)v36 )
            v36 = v35;
          goto LABEL_24;
        }
        if ( v104 == *(_QWORD *)v36 )
          goto LABEL_24;
        v36 += 8;
      }
      if ( v104 == *(_QWORD *)v36 )
        goto LABEL_24;
      v36 += 8;
      goto LABEL_115;
    }
  }
  v14 = *(_QWORD *)(v12 + 48);
  if ( !v14 )
    BUG();
  if ( *(_BYTE *)(v14 - 8) != 77 )
    return result;
  result = *(_DWORD *)(v14 - 4) & 0xFFFFFFF;
  if ( (*(_DWORD *)(v14 - 4) & 0xFFFFFFF) == 0 )
    return result;
  v15 = *(_DWORD *)(a1 + 64);
  v106 = v12;
  v16 = 0;
  v17 = *(_QWORD *)(a1 + 48);
  v18 = v14 - 24 - 24LL * (unsigned int)result;
  v19 = v15 - 1;
  v20 = 24LL * *(unsigned int *)(v14 + 32);
  v21 = v20 + 8LL * (unsigned int)(result - 1) + 16;
  v22 = v20 + 8;
  v23 = 0;
  do
  {
    while ( 1 )
    {
      result = v18;
      if ( (*(_BYTE *)(v14 - 1) & 0x40) != 0 )
        result = *(_QWORD *)(v14 - 32);
      if ( !v15 )
        goto LABEL_12;
      v24 = *(_QWORD *)(result + v22);
      result = v19 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v25 = *(_QWORD *)(v17 + 8 * result);
      if ( v24 != v25 )
        break;
LABEL_8:
      v22 += 8;
      ++v23;
      if ( v22 == v21 )
        goto LABEL_13;
    }
    v105 = 1;
    while ( v25 != -8 )
    {
      result = v19 & (unsigned int)(v105 + result);
      v25 = *(_QWORD *)(v17 + 8LL * (unsigned int)result);
      if ( v24 == v25 )
        goto LABEL_8;
      ++v105;
    }
LABEL_12:
    v22 += 8;
    ++v16;
  }
  while ( v22 != v21 );
LABEL_13:
  v10 = a1;
  v103 = v23;
  if ( v16 > 1 )
  {
    v26 = *(_QWORD *)a1;
    v27 = a1 + 40;
    v28 = sub_157ED20(v106);
    v113 = sub_1AA8CA0((_QWORD *)*a2, v28, v26, 0);
    v104 = *a2;
    v29 = *(_DWORD *)(a1 + 64);
    if ( v29 )
    {
      v30 = *(_QWORD *)(a1 + 48);
      goto LABEL_16;
    }
LABEL_27:
    sub_1ABFB80(v27, &v113);
    result = v113;
    *a2 = v113;
    if ( v103 )
    {
      v40 = *(_QWORD *)(v104 + 48);
      if ( !v40 )
        BUG();
      result = *(_DWORD *)(v40 - 4) & 0xFFFFFFF;
      if ( (*(_DWORD *)(v40 - 4) & 0xFFFFFFF) != 0 )
      {
        v41 = v40 - 24;
        v42 = 0;
        v43 = *(_QWORD *)(v104 + 48);
        v44 = 8LL * (unsigned int)result;
        while ( 1 )
        {
          if ( (*(_BYTE *)(v43 - 1) & 0x40) != 0 )
            v45 = *(_QWORD *)(v43 - 32);
          else
            v45 = v41 - 24LL * (*(_DWORD *)(v43 - 4) & 0xFFFFFFF);
          v46 = *(_DWORD *)(v10 + 64);
          if ( !v46 )
            goto LABEL_35;
          v47 = *(_QWORD *)(v10 + 48);
          v48 = 1;
          v49 = *(_QWORD *)(v42 + v45 + 24LL * *(unsigned int *)(v43 + 32) + 8);
          v50 = v46 - 1;
          v51 = (v46 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
          v52 = *(_QWORD *)(v47 + 8LL * v51);
          if ( v49 == v52 )
          {
LABEL_34:
            v53 = sub_157EBA0(v49);
            sub_1648780(v53, v104, v113);
LABEL_35:
            v42 += 8;
            if ( v44 == v42 )
              goto LABEL_96;
          }
          else
          {
            while ( v52 != -8 )
            {
              v51 = v50 & (v48 + v51);
              v52 = *(_QWORD *)(v47 + 8LL * v51);
              if ( v49 == v52 )
                goto LABEL_34;
              ++v48;
            }
            v42 += 8;
            if ( v44 == v42 )
            {
LABEL_96:
              result = v104;
              v40 = *(_QWORD *)(v104 + 48);
              break;
            }
          }
        }
      }
      while ( 2 )
      {
        if ( !v40 )
          BUG();
        if ( *(_BYTE *)(v40 - 8) == 77 )
        {
          v54 = *(_QWORD *)(v113 + 48);
          if ( v54 )
            v54 -= 24;
          v107 = v54;
          v114[0] = sub_1649960(v40 - 24);
          v116 = 773;
          v115[0] = (__int64)v114;
          v115[1] = (__int64)".ce";
          v114[1] = v55;
          v110 = *(_QWORD *)(v40 - 24);
          v56 = sub_1648B60(64);
          v59 = v107;
          v60 = v56;
          if ( v56 )
          {
            v108 = v56;
            sub_15F1EA0(v56, v110, 53, 0, 0, v59);
            *(_DWORD *)(v60 + 56) = v103 + 1;
            sub_164B780(v60, v115);
            sub_1648880(v60, *(_DWORD *)(v60 + 56), 1);
          }
          else
          {
            v108 = 0;
          }
          sub_164D160(v40 - 24, v60, a3, a4, a5, a6, v57, v58, a9, a10);
          v65 = *(_DWORD *)(v60 + 20) & 0xFFFFFFF;
          if ( v65 == *(_DWORD *)(v60 + 56) )
          {
            sub_15F55D0(v60, v60, v61, v62, v63, v64);
            v65 = *(_DWORD *)(v60 + 20) & 0xFFFFFFF;
          }
          v66 = (v65 + 1) & 0xFFFFFFF;
          v67 = v66 | *(_DWORD *)(v60 + 20) & 0xF0000000;
          *(_DWORD *)(v60 + 20) = v67;
          if ( (v67 & 0x40000000) != 0 )
            v68 = *(_QWORD *)(v60 - 8);
          else
            v68 = v108 - 24 * v66;
          v69 = (_QWORD *)(v68 + 24LL * (unsigned int)(v66 - 1));
          if ( *v69 )
          {
            v70 = v69[1];
            v71 = v69[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v71 = v70;
            if ( v70 )
              *(_QWORD *)(v70 + 16) = *(_QWORD *)(v70 + 16) & 3LL | v71;
          }
          *v69 = v40 - 24;
          v72 = *(_QWORD *)(v40 - 16);
          v69[1] = v72;
          if ( v72 )
            *(_QWORD *)(v72 + 16) = (unsigned __int64)(v69 + 1) | *(_QWORD *)(v72 + 16) & 3LL;
          v69[2] = (v40 - 16) | v69[2] & 3LL;
          *(_QWORD *)(v40 - 16) = v69;
          v73 = *(_DWORD *)(v60 + 20) & 0xFFFFFFF;
          if ( (*(_BYTE *)(v60 + 23) & 0x40) != 0 )
            v74 = *(_QWORD *)(v60 - 8);
          else
            v74 = v108 - 24 * v73;
          v75 = v40 - 24;
          v76 = v60;
          *(_QWORD *)(v74 + 8LL * (unsigned int)(v73 - 1) + 24LL * *(unsigned int *)(v60 + 56) + 8) = v104;
          v77 = *(unsigned int *)(v40 - 4);
          v78 = 0;
          result = *(_DWORD *)(v40 - 4) & 0xFFFFFFF;
          if ( (*(_DWORD *)(v40 - 4) & 0xFFFFFFF) == 0 )
          {
LABEL_40:
            v40 = *(_QWORD *)(v40 + 8);
            continue;
          }
          while ( 2 )
          {
            if ( (*(_BYTE *)(v40 - 1) & 0x40) != 0 )
              v79 = *(_QWORD *)(v40 - 32);
            else
              v79 = v75 - 24LL * (unsigned int)result;
            v80 = *(_DWORD *)(v10 + 64);
            if ( v80 )
            {
              v81 = (unsigned int)(v80 - 1);
              v82 = *(_QWORD *)(v10 + 48);
              v83 = 1;
              v84 = *(_QWORD *)(v79 + 8LL * v78 + 24LL * *(unsigned int *)(v40 + 32) + 8);
              v85 = v81 & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
              v86 = *(_QWORD *)(v82 + 8LL * v85);
              if ( v84 == v86 )
              {
LABEL_63:
                v87 = *(_QWORD *)(v79 + 24LL * v78);
                v88 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
                if ( v88 == *(_DWORD *)(v76 + 56) )
                {
                  v102 = v75;
                  v112 = v87;
                  sub_15F55D0(v76, v78, v81, v87, v77, v75);
                  v75 = v102;
                  v87 = v112;
                  v88 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
                }
                v89 = (v88 + 1) & 0xFFFFFFF;
                v90 = v89 | *(_DWORD *)(v76 + 20) & 0xF0000000;
                *(_DWORD *)(v76 + 20) = v90;
                if ( (v90 & 0x40000000) != 0 )
                  v91 = *(_QWORD *)(v76 - 8);
                else
                  v91 = v108 - 24 * v89;
                v92 = (__int64 *)(v91 + 24LL * (unsigned int)(v89 - 1));
                if ( *v92 )
                {
                  v93 = v92[1];
                  v94 = v92[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v94 = v93;
                  if ( v93 )
                    *(_QWORD *)(v93 + 16) = *(_QWORD *)(v93 + 16) & 3LL | v94;
                }
                *v92 = v87;
                if ( v87 )
                {
                  v95 = *(_QWORD *)(v87 + 8);
                  v92[1] = v95;
                  if ( v95 )
                    *(_QWORD *)(v95 + 16) = (unsigned __int64)(v92 + 1) | *(_QWORD *)(v95 + 16) & 3LL;
                  v92[2] = (v87 + 8) | v92[2] & 3;
                  *(_QWORD *)(v87 + 8) = v92;
                }
                v96 = *(_DWORD *)(v76 + 20) & 0xFFFFFFF;
                if ( (*(_BYTE *)(v76 + 23) & 0x40) != 0 )
                  v97 = *(_QWORD *)(v76 - 8);
                else
                  v97 = v108 - 24 * v96;
                v111 = v75;
                *(_QWORD *)(v97 + 8LL * (unsigned int)(v96 - 1) + 24LL * *(unsigned int *)(v76 + 56) + 8) = v84;
                sub_15F5350(v75, v78, 1);
                v77 = *(unsigned int *)(v40 - 4);
                v75 = v111;
LABEL_77:
                result = v77 & 0xFFFFFFF;
                if ( (_DWORD)result == v78 )
                  goto LABEL_40;
                continue;
              }
              while ( v86 != -8 )
              {
                v85 = v81 & (v83 + v85);
                v86 = *(_QWORD *)(v82 + 8LL * v85);
                if ( v84 == v86 )
                  goto LABEL_63;
                ++v83;
              }
            }
            break;
          }
          ++v78;
          goto LABEL_77;
        }
        break;
      }
    }
  }
  return result;
}
