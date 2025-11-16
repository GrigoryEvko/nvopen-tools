// Function: sub_13CA600
// Address: 0x13ca600
//
__int64 __fastcall sub_13CA600(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 *v7; // rax
  unsigned int v8; // r15d
  char v9; // dl
  _QWORD *v11; // rdi
  unsigned __int64 v12; // rcx
  unsigned __int8 *v13; // rax
  unsigned __int8 *v14; // rsi
  _QWORD *v15; // rdx
  _QWORD *v16; // rax
  _QWORD *v17; // r14
  __int64 v18; // rax
  _BYTE *v19; // r14
  __int64 v20; // rax
  _BYTE *v21; // r15
  char v22; // dl
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rsi
  unsigned int v28; // ecx
  __int64 *v29; // rdx
  __int64 v30; // r9
  __int64 *v31; // r14
  __int64 v32; // r12
  int v33; // eax
  __int64 v34; // rsi
  int v35; // ecx
  __int64 v36; // rdi
  unsigned int v37; // edx
  __int64 *v38; // rax
  __int64 v39; // r8
  __int64 v40; // r13
  _QWORD *v41; // rdx
  _QWORD *v42; // rax
  _QWORD *v43; // rcx
  __int64 v44; // rax
  _QWORD *v45; // rax
  int v46; // ecx
  __int64 v47; // rsi
  __int64 v48; // rax
  __int64 v49; // rdi
  int v50; // ecx
  __int64 v51; // r8
  unsigned int v52; // edx
  __int64 *v53; // rax
  __int64 v54; // r10
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // r15
  __int64 v58; // rax
  __int64 v59; // r14
  unsigned __int64 v60; // r8
  __int64 *v61; // rdi
  unsigned int v62; // r8d
  __int64 *v63; // rcx
  __int64 v64; // rdx
  __int64 v65; // rcx
  _QWORD *v66; // rdx
  int v67; // eax
  int v68; // r9d
  __int64 v69; // r8
  _QWORD *v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // rsi
  _QWORD *v73; // rdi
  _QWORD *v74; // rax
  _QWORD *v75; // rcx
  _QWORD *v76; // rsi
  unsigned int v77; // edi
  _QWORD *v78; // rcx
  __int64 v79; // r14
  unsigned int v80; // eax
  __int64 v81; // rcx
  _QWORD *v82; // rbx
  unsigned __int64 *v83; // rcx
  unsigned __int64 v84; // rdx
  unsigned __int64 v85; // rdi
  __int64 v86; // rax
  __int64 v87; // rax
  int i; // edx
  int v89; // r8d
  int v90; // eax
  int v91; // r9d
  unsigned __int8 v92; // [rsp+7h] [rbp-E9h]
  __int64 v93; // [rsp+8h] [rbp-E8h]
  __int64 v94; // [rsp+18h] [rbp-D8h]
  _QWORD *v95; // [rsp+20h] [rbp-D0h]
  _QWORD *v96; // [rsp+28h] [rbp-C8h]
  __int64 v97; // [rsp+30h] [rbp-C0h]
  _QWORD *v98; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v99; // [rsp+48h] [rbp-A8h] BYREF
  _QWORD v100[4]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v101; // [rsp+70h] [rbp-80h] BYREF
  _BYTE *v102; // [rsp+78h] [rbp-78h]
  _BYTE *v103; // [rsp+80h] [rbp-70h]
  __int64 v104; // [rsp+88h] [rbp-68h]
  int v105; // [rsp+90h] [rbp-60h]
  _BYTE v106[88]; // [rsp+98h] [rbp-58h] BYREF

  v98 = (_QWORD *)a2;
  v5 = sub_15F2050(a2);
  v6 = sub_1632FA0(v5);
  v7 = *(__int64 **)(a1 + 48);
  if ( *(__int64 **)(a1 + 56) != v7 )
    goto LABEL_2;
  v61 = &v7[*(unsigned int *)(a1 + 68)];
  v62 = *(_DWORD *)(a1 + 68);
  if ( v7 == v61 )
    goto LABEL_70;
  v63 = 0;
  do
  {
    if ( v98 == (_QWORD *)*v7 )
      return 1;
    if ( *v7 == -2 )
      v63 = v7;
    ++v7;
  }
  while ( v61 != v7 );
  if ( v63 )
  {
    *v63 = (__int64)v98;
    --*(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_6;
  }
LABEL_70:
  if ( v62 >= *(_DWORD *)(a1 + 64) )
  {
LABEL_2:
    v8 = 1;
    sub_16CCBA0(a1 + 40, a2);
    if ( v9 )
      goto LABEL_6;
    return v8;
  }
  *(_DWORD *)(a1 + 68) = v62 + 1;
  *v61 = a2;
  ++*(_QWORD *)(a1 + 40);
LABEL_6:
  if ( !(unsigned __int8)sub_1456C80(*(_QWORD *)(a1 + 32), *v98) )
    return 0;
  v11 = v98;
  if ( *((_BYTE *)v98 + 16) == 77 )
  {
LABEL_10:
    v12 = sub_1456C90(*(_QWORD *)(a1 + 32), *v11);
    if ( v12 > 0x40 )
      return 0;
    v13 = *(unsigned __int8 **)(v6 + 24);
    v14 = &v13[*(unsigned int *)(v6 + 32)];
    if ( v14 == v13 )
      return 0;
    while ( *v13 != v12 )
    {
      if ( v14 == ++v13 )
        return 0;
    }
    v15 = *(_QWORD **)(a1 + 240);
    v16 = *(_QWORD **)(a1 + 232);
    if ( v15 == v16 )
    {
      v17 = &v16[*(unsigned int *)(a1 + 252)];
      if ( v16 == v17 )
      {
        v66 = *(_QWORD **)(a1 + 232);
      }
      else
      {
        do
        {
          if ( v98 == (_QWORD *)*v16 )
            break;
          ++v16;
        }
        while ( v17 != v16 );
        v66 = v17;
      }
    }
    else
    {
      v17 = &v15[*(unsigned int *)(a1 + 248)];
      v16 = (_QWORD *)sub_16CC9F0(a1 + 224, v98);
      if ( v98 == (_QWORD *)*v16 )
      {
        v64 = *(_QWORD *)(a1 + 240);
        if ( v64 == *(_QWORD *)(a1 + 232) )
          v65 = *(unsigned int *)(a1 + 252);
        else
          v65 = *(unsigned int *)(a1 + 248);
        v66 = (_QWORD *)(v64 + 8 * v65);
      }
      else
      {
        v18 = *(_QWORD *)(a1 + 240);
        if ( v18 != *(_QWORD *)(a1 + 232) )
        {
          v16 = (_QWORD *)(v18 + 8LL * *(unsigned int *)(a1 + 248));
LABEL_19:
          if ( v17 != v16 )
            return 0;
          v94 = sub_146F1B0(*(_QWORD *)(a1 + 32), v98);
          v8 = sub_13C9880(v94, (__int64)v98, *(_QWORD *)a1, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16));
          if ( !(_BYTE)v8 )
            return 0;
          v19 = v106;
          v104 = 4;
          v102 = v106;
          v103 = v106;
          v105 = 0;
          v20 = v98[1];
          v101 = 0;
          v97 = v20;
          if ( !v20 )
            return v8;
          v93 = a1 + 40;
          v96 = (_QWORD *)a1;
          v92 = v8;
          v21 = v106;
          while ( 1 )
          {
            v99 = sub_1648700(v97);
            if ( v19 == v21 )
            {
              v73 = &v19[8 * HIDWORD(v104)];
              if ( v73 != (_QWORD *)v19 )
              {
                v74 = v19;
                v75 = 0;
                do
                {
                  if ( v99 == *v74 )
                    goto LABEL_58;
                  if ( *v74 == -2 )
                    v75 = v74;
                  ++v74;
                }
                while ( v73 != v74 );
                if ( v75 )
                {
                  *v75 = v99;
                  --v105;
                  ++v101;
                  goto LABEL_25;
                }
              }
              if ( HIDWORD(v104) < (unsigned int)v104 )
                break;
            }
            sub_16CCBA0(&v101, v99);
            v21 = v103;
            v19 = v102;
            if ( v22 )
              goto LABEL_25;
LABEL_58:
            v97 = *(_QWORD *)(v97 + 8);
            if ( !v97 )
            {
              v60 = (unsigned __int64)v21;
              v8 = v92;
LABEL_60:
              if ( v19 != (_BYTE *)v60 )
                _libc_free(v60);
              return v8;
            }
          }
          ++HIDWORD(v104);
          *v73 = v99;
          ++v101;
LABEL_25:
          if ( *(_BYTE *)(v99 + 16) == 77 )
          {
            if ( sub_13A0E30(v93, v99) )
              goto LABEL_57;
            v79 = v99;
            if ( *(_BYTE *)(v99 + 16) == 77 )
            {
              v80 = sub_1648720(v97);
              if ( (*(_BYTE *)(v79 + 23) & 0x40) != 0 )
                v81 = *(_QWORD *)(v79 - 8);
              else
                v81 = v79 - 24LL * (*(_DWORD *)(v79 + 20) & 0xFFFFFFF);
              v23 = *(_QWORD *)(v81 + 8LL * v80 + 24LL * *(unsigned int *)(v79 + 56) + 8);
            }
            else
            {
              v23 = *(_QWORD *)(v99 + 40);
            }
          }
          else
          {
            v23 = *(_QWORD *)(v99 + 40);
          }
          v24 = v96[3];
          v25 = v96[2];
          v26 = *(unsigned int *)(v24 + 48);
          if ( !(_DWORD)v26 )
            goto LABEL_47;
          v27 = *(_QWORD *)(v24 + 32);
          v28 = (v26 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v29 = (__int64 *)(v27 + 16LL * v28);
          v30 = *v29;
          if ( v23 != *v29 )
          {
            for ( i = 1; ; i = v89 )
            {
              if ( v30 == -8 )
                goto LABEL_47;
              v89 = i + 1;
              v28 = (v26 - 1) & (i + v28);
              v29 = (__int64 *)(v27 + 16LL * v28);
              v30 = *v29;
              if ( v23 == *v29 )
                break;
            }
          }
          if ( v29 != (__int64 *)(v27 + 16 * v26) )
          {
            v31 = (__int64 *)v29[1];
            if ( v31 )
            {
              v32 = 0;
              while ( 2 )
              {
                v33 = *(_DWORD *)(v25 + 24);
                if ( !v33 )
                  goto LABEL_32;
                v34 = *v31;
                v35 = v33 - 1;
                v36 = *(_QWORD *)(v25 + 8);
                v37 = (v33 - 1) & (((unsigned int)*v31 >> 9) ^ ((unsigned int)*v31 >> 4));
                v38 = (__int64 *)(v36 + 16LL * v37);
                v39 = *v38;
                if ( *v31 != *v38 )
                {
                  v67 = 1;
                  if ( v39 == -8 )
                    goto LABEL_32;
                  while ( 1 )
                  {
                    v68 = v67 + 1;
                    v37 = v35 & (v67 + v37);
                    v38 = (__int64 *)(v36 + 16LL * v37);
                    v69 = *v38;
                    if ( v34 == *v38 )
                      break;
                    v67 = v68;
                    if ( v69 == -8 )
                      goto LABEL_32;
                  }
                }
                v40 = v38[1];
                if ( !v40 || v34 != **(_QWORD **)(v40 + 32) )
                  goto LABEL_32;
                if ( !(unsigned __int8)sub_13FCBF0(v40) )
                  goto LABEL_139;
                v41 = *(_QWORD **)(a3 + 16);
                v42 = *(_QWORD **)(a3 + 8);
                if ( v41 == v42 )
                {
                  v70 = &v42[*(unsigned int *)(a3 + 28)];
                  if ( v42 == v70 )
                  {
                    v43 = *(_QWORD **)(a3 + 8);
                  }
                  else
                  {
                    do
                    {
                      if ( v40 == *v42 )
                        break;
                      ++v42;
                    }
                    while ( v70 != v42 );
                    v43 = v70;
                  }
                }
                else
                {
                  v95 = &v41[*(unsigned int *)(a3 + 24)];
                  v42 = (_QWORD *)sub_16CC9F0(a3, v40);
                  v43 = v95;
                  if ( v40 == *v42 )
                  {
                    v71 = *(_QWORD *)(a3 + 16);
                    if ( v71 == *(_QWORD *)(a3 + 8) )
                      v72 = *(unsigned int *)(a3 + 28);
                    else
                      v72 = *(unsigned int *)(a3 + 24);
                    v70 = (_QWORD *)(v71 + 8 * v72);
                  }
                  else
                  {
                    v44 = *(_QWORD *)(a3 + 16);
                    if ( v44 != *(_QWORD *)(a3 + 8) )
                    {
                      v42 = (_QWORD *)(v44 + 8LL * *(unsigned int *)(a3 + 24));
                      goto LABEL_42;
                    }
                    v42 = (_QWORD *)(v44 + 8LL * *(unsigned int *)(a3 + 28));
                    v70 = v42;
                  }
                }
                while ( v70 != v42 && *v42 >= 0xFFFFFFFFFFFFFFFELL )
                  ++v42;
LABEL_42:
                if ( v42 != v43 )
                {
LABEL_43:
                  if ( v32 )
                  {
                    v45 = *(_QWORD **)(a3 + 8);
                    if ( *(_QWORD **)(a3 + 16) != v45 )
                      goto LABEL_45;
                    v76 = &v45[*(unsigned int *)(a3 + 28)];
                    v77 = *(_DWORD *)(a3 + 28);
                    if ( v45 == v76 )
                    {
LABEL_148:
                      if ( v77 >= *(_DWORD *)(a3 + 24) )
                      {
LABEL_45:
                        sub_16CCBA0(a3, v32);
                      }
                      else
                      {
                        *(_DWORD *)(a3 + 28) = v77 + 1;
                        *v76 = v32;
                        ++*(_QWORD *)a3;
                      }
                    }
                    else
                    {
                      v78 = 0;
                      while ( *v45 != v32 )
                      {
                        if ( *v45 == -2 )
                          v78 = v45;
                        if ( v76 == ++v45 )
                        {
                          if ( !v78 )
                            goto LABEL_148;
                          *v78 = v32;
                          --*(_DWORD *)(a3 + 32);
                          ++*(_QWORD *)a3;
                          break;
                        }
                      }
                    }
                  }
                  v25 = v96[2];
                  break;
                }
                if ( !v32 )
                  v32 = v40;
LABEL_32:
                v31 = (__int64 *)v31[1];
                if ( !v31 )
                  goto LABEL_43;
                continue;
              }
            }
          }
LABEL_47:
          v46 = *(_DWORD *)(v25 + 24);
          v47 = v99;
          v48 = 0;
          if ( v46 )
          {
            v49 = *(_QWORD *)(v99 + 40);
            v50 = v46 - 1;
            v51 = *(_QWORD *)(v25 + 8);
            v52 = v50 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
            v53 = (__int64 *)(v51 + 16LL * v52);
            v54 = *v53;
            if ( v49 == *v53 )
            {
LABEL_49:
              v48 = v53[1];
            }
            else
            {
              v90 = 1;
              while ( v54 != -8 )
              {
                v91 = v90 + 1;
                v52 = v50 & (v90 + v52);
                v53 = (__int64 *)(v51 + 16LL * v52);
                v54 = *v53;
                if ( v49 == *v53 )
                  goto LABEL_49;
                v90 = v91;
              }
              v48 = 0;
            }
          }
          if ( *v96 != v48 && *(_BYTE *)(v99 + 16) == 77 )
            goto LABEL_54;
          if ( sub_13A0E30(v93, v99) || !(unsigned __int8)sub_13CA600(v96, v99, a3) )
          {
            v47 = v99;
LABEL_54:
            v55 = sub_13C9CF0((__int64)v96, v47, (__int64)v98);
            v56 = v96[4];
            v57 = v55;
            v100[2] = v96;
            v100[3] = v55;
            v100[0] = &v99;
            v100[1] = &v98;
            v58 = sub_14999C0(v94, sub_13C95C0, v100, v56);
            v59 = v58;
            if ( v58 != v94 && v94 != sub_1499A20(v58, v57 + 80, v96[4]) )
            {
              v82 = (_QWORD *)(v96[26] & 0xFFFFFFFFFFFFFFF8LL);
              v83 = (unsigned __int64 *)v82[1];
              v84 = *v82 & 0xFFFFFFFFFFFFFFF8LL;
              *v83 = v84 | *v83 & 7;
              *(_QWORD *)(v84 + 8) = v83;
              v85 = v82[8];
              *v82 &= 7uLL;
              v82[1] = 0;
              *(v82 - 4) = &unk_49EA628;
              if ( v85 != v82[7] )
                _libc_free(v85);
              v86 = v82[5];
              if ( v86 != 0 && v86 != -8 && v86 != -16 )
                sub_1649B30(v82 + 3);
              *(v82 - 4) = &unk_49EE2B0;
              v87 = *(v82 - 1);
              if ( v87 != 0 && v87 != -8 && v87 != -16 )
                sub_1649B30(v82 - 3);
              j_j___libc_free_0(v82 - 4, 136);
LABEL_139:
              v60 = (unsigned __int64)v103;
              v19 = v102;
              v8 = 0;
              goto LABEL_60;
            }
            v94 = v59;
          }
LABEL_57:
          v21 = v103;
          v19 = v102;
          goto LABEL_58;
        }
        v16 = (_QWORD *)(v18 + 8LL * *(unsigned int *)(a1 + 252));
        v66 = v16;
      }
    }
    while ( v66 != v16 && *v16 >= 0xFFFFFFFFFFFFFFFELL )
      ++v16;
    goto LABEL_19;
  }
  if ( (unsigned __int8)sub_14AF470(v98, 0, 0, 0) )
  {
    v11 = v98;
    goto LABEL_10;
  }
  return 0;
}
