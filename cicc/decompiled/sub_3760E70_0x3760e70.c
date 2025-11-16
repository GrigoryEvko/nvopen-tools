// Function: sub_3760E70
// Address: 0x3760e70
//
__int64 __fastcall sub_3760E70(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  int v8; // r12d
  int v9; // eax
  char v10; // cl
  __int64 v11; // r9
  int v12; // esi
  unsigned int v13; // edi
  _DWORD *v14; // rdx
  int v15; // r10d
  _DWORD *v16; // rdx
  int *v17; // r8
  __int64 v18; // r9
  int v19; // eax
  __int64 v20; // rcx
  __int64 v21; // r12
  __int64 v22; // rdx
  int **v23; // rsi
  unsigned int v24; // r14d
  __int64 v25; // r8
  unsigned __int64 v26; // r12
  int v27; // r13d
  int v28; // eax
  char v29; // cl
  int v30; // esi
  unsigned int v31; // edi
  int *v32; // rdx
  int v33; // r10d
  _DWORD *v34; // rdx
  unsigned int v35; // esi
  __int64 v36; // r8
  int v37; // ecx
  unsigned int v38; // edi
  __int64 v39; // rax
  int v40; // r10d
  __int64 v41; // rax
  unsigned int v42; // edx
  int v43; // edi
  unsigned int v44; // r10d
  _BYTE *v45; // rdi
  int v47; // esi
  int v48; // r15d
  unsigned int v49; // esi
  __int64 v50; // rsi
  int v51; // edx
  unsigned int v52; // ecx
  int v53; // edi
  __int64 v54; // rax
  __int64 v55; // rsi
  int v56; // edx
  unsigned int v57; // ecx
  int v58; // edi
  int v59; // r11d
  int v60; // eax
  int v61; // edx
  int v62; // edx
  unsigned int v63; // edi
  _DWORD *v64; // r8
  int v65; // r9d
  unsigned int v66; // r11d
  int v67; // r11d
  int v68; // edx
  __int64 v69; // r10
  int v70; // r9d
  unsigned int v71; // edx
  int v72; // edi
  __int64 v73; // r9
  int v74; // edx
  unsigned int v75; // ecx
  int v76; // edi
  int v77; // esi
  _DWORD *v78; // r10
  int v79; // r11d
  int v80; // r9d
  int v81; // edx
  int v82; // esi
  _DWORD *v83; // rcx
  int v86; // [rsp+1Ch] [rbp-154h]
  __int64 *v87; // [rsp+20h] [rbp-150h]
  unsigned int v88; // [rsp+28h] [rbp-148h]
  __int64 v89; // [rsp+30h] [rbp-140h]
  int v90; // [rsp+30h] [rbp-140h]
  unsigned __int64 v91; // [rsp+38h] [rbp-138h]
  int v92; // [rsp+38h] [rbp-138h]
  int v93; // [rsp+38h] [rbp-138h]
  unsigned __int64 v94; // [rsp+40h] [rbp-130h] BYREF
  __int64 v95; // [rsp+48h] [rbp-128h]
  int v96; // [rsp+5Ch] [rbp-114h] BYREF
  __int64 (__fastcall **v97)(); // [rsp+60h] [rbp-110h] BYREF
  __int64 v98; // [rsp+68h] [rbp-108h]
  __int64 v99; // [rsp+70h] [rbp-100h]
  __int64 v100; // [rsp+78h] [rbp-F8h]
  __int64 *v101; // [rsp+80h] [rbp-F0h]
  __int64 v102; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v103; // [rsp+98h] [rbp-D8h]
  __int64 v104; // [rsp+A0h] [rbp-D0h]
  __int64 v105; // [rsp+A8h] [rbp-C8h]
  _BYTE *v106; // [rsp+B0h] [rbp-C0h]
  __int64 v107; // [rsp+B8h] [rbp-B8h]
  _BYTE v108[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v94 = a4;
  v95 = a5;
  sub_375EAB0(a1, (__int64)&v94);
  v102 = 0;
  v106 = v108;
  v107 = 0x1000000000LL;
  v6 = *(_QWORD *)(a1 + 8);
  v103 = 0;
  v7 = *(_QWORD *)(v6 + 768);
  v99 = v6;
  v104 = 0;
  v98 = v7;
  *(_QWORD *)(v6 + 768) = &v97;
  v105 = 0;
  v97 = off_4A3D440;
  v100 = a1;
  v101 = &v102;
  do
  {
    v8 = sub_375D5B0(a1, a2, a3);
    v9 = sub_375D5B0(a1, v94, v95);
    if ( v8 == v9 )
      goto LABEL_8;
    v10 = *(_BYTE *)(a1 + 1536) & 1;
    if ( v10 )
    {
      v11 = a1 + 1544;
      v12 = 7;
    }
    else
    {
      v49 = *(_DWORD *)(a1 + 1552);
      v11 = *(_QWORD *)(a1 + 1544);
      if ( !v49 )
      {
        v63 = *(_DWORD *)(a1 + 1536);
        ++*(_QWORD *)(a1 + 1528);
        v64 = 0;
        v65 = (v63 >> 1) + 1;
        goto LABEL_81;
      }
      v12 = v49 - 1;
    }
    v13 = v12 & (37 * v8);
    v14 = (_DWORD *)(v11 + 8LL * v13);
    v15 = *v14;
    if ( v8 == *v14 )
      goto LABEL_6;
    v67 = 1;
    v64 = 0;
    while ( 1 )
    {
      if ( v15 == -1 )
      {
        v63 = *(_DWORD *)(a1 + 1536);
        v66 = 24;
        v49 = 8;
        if ( !v64 )
          v64 = v14;
        ++*(_QWORD *)(a1 + 1528);
        v65 = (v63 >> 1) + 1;
        if ( v10 )
        {
LABEL_82:
          if ( v66 > 4 * v65 )
          {
            if ( v49 - *(_DWORD *)(a1 + 1540) - v65 > v49 >> 3 )
            {
LABEL_84:
              *(_DWORD *)(a1 + 1536) = (2 * (v63 >> 1) + 2) | v63 & 1;
              if ( *v64 != -1 )
                --*(_DWORD *)(a1 + 1540);
              *v64 = v8;
              v16 = v64 + 1;
              v64[1] = 0;
              goto LABEL_7;
            }
            v93 = v9;
            sub_375BDE0(a1 + 1528, v49);
            v9 = v93;
            if ( (*(_BYTE *)(a1 + 1536) & 1) != 0 )
            {
              v73 = a1 + 1544;
              v74 = 7;
              goto LABEL_106;
            }
            v81 = *(_DWORD *)(a1 + 1552);
            v73 = *(_QWORD *)(a1 + 1544);
            if ( v81 )
            {
              v74 = v81 - 1;
LABEL_106:
              v75 = v74 & (37 * v8);
              v64 = (_DWORD *)(v73 + 8LL * v75);
              v76 = *v64;
              if ( v8 != *v64 )
              {
                v77 = 1;
                v78 = 0;
                while ( v76 != -1 )
                {
                  if ( v76 == -2 && !v78 )
                    v78 = v64;
                  v75 = v74 & (v77 + v75);
                  v64 = (_DWORD *)(v73 + 8LL * v75);
                  v76 = *v64;
                  if ( v8 == *v64 )
                    goto LABEL_102;
                  ++v77;
                }
                if ( v78 )
                  v64 = v78;
              }
LABEL_102:
              v63 = *(_DWORD *)(a1 + 1536);
              goto LABEL_84;
            }
LABEL_148:
            *(_DWORD *)(a1 + 1536) = (2 * (*(_DWORD *)(a1 + 1536) >> 1) + 2) | *(_DWORD *)(a1 + 1536) & 1;
            BUG();
          }
          v92 = v9;
          sub_375BDE0(a1 + 1528, 2 * v49);
          v9 = v92;
          if ( (*(_BYTE *)(a1 + 1536) & 1) != 0 )
          {
            v69 = a1 + 1544;
            v70 = 7;
          }
          else
          {
            v68 = *(_DWORD *)(a1 + 1552);
            v69 = *(_QWORD *)(a1 + 1544);
            if ( !v68 )
              goto LABEL_148;
            v70 = v68 - 1;
          }
          v71 = v70 & (37 * v8);
          v64 = (_DWORD *)(v69 + 8LL * v71);
          v72 = *v64;
          if ( v8 != *v64 )
          {
            v82 = 1;
            v83 = 0;
            while ( v72 != -1 )
            {
              if ( !v83 && v72 == -2 )
                v83 = v64;
              v71 = v70 & (v82 + v71);
              v64 = (_DWORD *)(v69 + 8LL * v71);
              v72 = *v64;
              if ( v8 == *v64 )
                goto LABEL_102;
              ++v82;
            }
            if ( v83 )
            {
              v63 = *(_DWORD *)(a1 + 1536);
              v64 = v83;
              goto LABEL_84;
            }
          }
          goto LABEL_102;
        }
        v49 = *(_DWORD *)(a1 + 1552);
LABEL_81:
        v66 = 3 * v49;
        goto LABEL_82;
      }
      if ( v64 || v15 != -2 )
        v14 = v64;
      v13 = v12 & (v67 + v13);
      v15 = *(_DWORD *)(v11 + 8LL * v13);
      if ( v8 == v15 )
        break;
      ++v67;
      v64 = v14;
      v14 = (_DWORD *)(v11 + 8LL * v13);
    }
    v14 = (_DWORD *)(v11 + 8LL * v13);
LABEL_6:
    v16 = v14 + 1;
LABEL_7:
    *v16 = v9;
LABEL_8:
    sub_34161C0(*(_QWORD *)(a1 + 8), a2, a3, v94, v95);
LABEL_9:
    v19 = v107;
    while ( v19 )
    {
      v20 = (__int64)v106;
      v21 = *(_QWORD *)&v106[8 * v19 - 8];
      v22 = (unsigned int)v105;
      if ( (_DWORD)v105 )
      {
        v20 = (unsigned int)(v105 - 1);
        v22 = (unsigned int)v20 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v23 = (int **)(v103 + 8 * v22);
        v17 = *v23;
        if ( (int *)v21 == *v23 )
        {
LABEL_13:
          *v23 = (int *)-8192LL;
          v19 = v107;
          LODWORD(v104) = v104 - 1;
          ++HIDWORD(v104);
        }
        else
        {
          v47 = 1;
          while ( v17 != (int *)-4096LL )
          {
            v18 = (unsigned int)(v47 + 1);
            v22 = (unsigned int)v20 & (v47 + (_DWORD)v22);
            v23 = (int **)(v103 + 8LL * (unsigned int)v22);
            v17 = *v23;
            if ( (int *)v21 == *v23 )
              goto LABEL_13;
            v47 = v18;
          }
        }
      }
      LODWORD(v107) = --v19;
      if ( *(_DWORD *)(v21 + 36) == -1 )
      {
        v87 = sub_375EBD0(a1, v21, v22, v20, (__int64)v17, v18);
        if ( (__int64 *)v21 == v87 )
          goto LABEL_9;
        v24 = 0;
        v86 = *(_DWORD *)(v21 + 68);
        if ( !v86 )
          goto LABEL_9;
        v91 = v21;
        while ( 2 )
        {
          v25 = v24;
          v26 = (unsigned __int64)v87;
          if ( *((_DWORD *)v87 + 9) != -3 )
          {
LABEL_19:
            v89 = v25;
            v27 = sub_375D5B0(a1, v91, v24);
            v88 = v89;
            v90 = sub_375D5B0(a1, v26, v89);
            sub_34161C0(*(_QWORD *)(a1 + 8), v91, v24, v26, v88);
            v28 = v90;
            if ( v27 == v90 )
              goto LABEL_25;
            v29 = *(_BYTE *)(a1 + 1536) & 1;
            if ( v29 )
            {
              v18 = a1 + 1544;
              v30 = 7;
              goto LABEL_22;
            }
            v35 = *(_DWORD *)(a1 + 1552);
            v18 = *(_QWORD *)(a1 + 1544);
            if ( !v35 )
            {
              v42 = *(_DWORD *)(a1 + 1536);
              ++*(_QWORD *)(a1 + 1528);
              v17 = 0;
              v43 = (v42 >> 1) + 1;
              goto LABEL_36;
            }
            v30 = v35 - 1;
LABEL_22:
            v31 = v30 & (37 * v27);
            v32 = (int *)(v18 + 8LL * v31);
            v33 = *v32;
            if ( v27 == *v32 )
            {
LABEL_23:
              v34 = v32 + 1;
              goto LABEL_24;
            }
            v48 = 1;
            v17 = 0;
            while ( v33 != -1 )
            {
              if ( v33 == -2 && !v17 )
                v17 = v32;
              v31 = v30 & (v48 + v31);
              v32 = (int *)(v18 + 8LL * v31);
              v33 = *v32;
              if ( v27 == *v32 )
                goto LABEL_23;
              ++v48;
            }
            v44 = 24;
            v35 = 8;
            if ( !v17 )
              v17 = v32;
            v42 = *(_DWORD *)(a1 + 1536);
            ++*(_QWORD *)(a1 + 1528);
            v43 = (v42 >> 1) + 1;
            if ( !v29 )
            {
              v35 = *(_DWORD *)(a1 + 1552);
LABEL_36:
              v44 = 3 * v35;
            }
            v18 = a1 + 1528;
            if ( 4 * v43 >= v44 )
            {
              sub_375BDE0(a1 + 1528, 2 * v35);
              v28 = v90;
              if ( (*(_BYTE *)(a1 + 1536) & 1) != 0 )
              {
                v50 = a1 + 1544;
                v51 = 7;
              }
              else
              {
                v61 = *(_DWORD *)(a1 + 1552);
                v50 = *(_QWORD *)(a1 + 1544);
                if ( !v61 )
                  goto LABEL_147;
                v51 = v61 - 1;
              }
              v52 = v51 & (37 * v27);
              v17 = (int *)(v50 + 8LL * v52);
              v53 = *v17;
              if ( v27 != *v17 )
              {
                v79 = 1;
                v18 = 0;
                while ( v53 != -1 )
                {
                  if ( !v18 && v53 == -2 )
                    v18 = (__int64)v17;
                  v52 = v51 & (v79 + v52);
                  v17 = (int *)(v50 + 8LL * v52);
                  v53 = *v17;
                  if ( v27 == *v17 )
                    goto LABEL_61;
                  ++v79;
                }
LABEL_70:
                if ( v18 )
                  v17 = (int *)v18;
              }
            }
            else
            {
              if ( v35 - *(_DWORD *)(a1 + 1540) - v43 > v35 >> 3 )
              {
LABEL_39:
                *(_DWORD *)(a1 + 1536) = (2 * (v42 >> 1) + 2) | v42 & 1;
                if ( *v17 != -1 )
                  --*(_DWORD *)(a1 + 1540);
                *v17 = v27;
                v34 = v17 + 1;
                v17[1] = 0;
LABEL_24:
                *v34 = v28;
LABEL_25:
                if ( ++v24 == v86 )
                  goto LABEL_9;
                continue;
              }
              sub_375BDE0(a1 + 1528, v35);
              v28 = v90;
              if ( (*(_BYTE *)(a1 + 1536) & 1) != 0 )
              {
                v55 = a1 + 1544;
                v56 = 7;
              }
              else
              {
                v62 = *(_DWORD *)(a1 + 1552);
                v55 = *(_QWORD *)(a1 + 1544);
                if ( !v62 )
                {
LABEL_147:
                  *(_DWORD *)(a1 + 1536) = (2 * (*(_DWORD *)(a1 + 1536) >> 1) + 2) | *(_DWORD *)(a1 + 1536) & 1;
                  BUG();
                }
                v56 = v62 - 1;
              }
              v57 = v56 & (37 * v27);
              v17 = (int *)(v55 + 8LL * v57);
              v58 = *v17;
              if ( v27 != *v17 )
              {
                v59 = 1;
                v18 = 0;
                while ( v58 != -1 )
                {
                  if ( !v18 && v58 == -2 )
                    v18 = (__int64)v17;
                  v57 = v56 & (v59 + v57);
                  v17 = (int *)(v55 + 8LL * v57);
                  v58 = *v17;
                  if ( v27 == *v17 )
                    goto LABEL_61;
                  ++v59;
                }
                goto LABEL_70;
              }
            }
LABEL_61:
            v42 = *(_DWORD *)(a1 + 1536);
            goto LABEL_39;
          }
          break;
        }
        v96 = sub_375D5B0(a1, (unsigned __int64)v87, v24);
        sub_37593F0(a1, &v96);
        if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
        {
          v36 = a1 + 520;
          v37 = 7;
          goto LABEL_31;
        }
        v41 = *(unsigned int *)(a1 + 528);
        v36 = *(_QWORD *)(a1 + 520);
        if ( (_DWORD)v41 )
        {
          v37 = v41 - 1;
LABEL_31:
          v38 = v37 & (37 * v96);
          v39 = v36 + 24LL * v38;
          v40 = *(_DWORD *)v39;
          if ( *(_DWORD *)v39 == v96 )
          {
LABEL_32:
            v26 = *(_QWORD *)(v39 + 8);
            v25 = *(unsigned int *)(v39 + 16);
            goto LABEL_19;
          }
          v60 = 1;
          while ( v40 != -1 )
          {
            v80 = v60 + 1;
            v38 = v37 & (v60 + v38);
            v39 = v36 + 24LL * v38;
            v40 = *(_DWORD *)v39;
            if ( v96 == *(_DWORD *)v39 )
              goto LABEL_32;
            v60 = v80;
          }
          if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
          {
            v54 = 192;
            goto LABEL_64;
          }
          v41 = *(unsigned int *)(a1 + 528);
        }
        v54 = 24 * v41;
LABEL_64:
        v39 = v36 + v54;
        goto LABEL_32;
      }
    }
  }
  while ( (unsigned __int8)sub_33CF8A0(a2, a3) );
  v45 = v106;
  *(_QWORD *)(v99 + 768) = v98;
  if ( v45 != v108 )
    _libc_free((unsigned __int64)v45);
  return sub_C7D6A0(v103, 8LL * (unsigned int)v105, 8);
}
