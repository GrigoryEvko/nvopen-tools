// Function: sub_BB68F0
// Address: 0xbb68f0
//
__int64 __fastcall sub_BB68F0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r15
  _QWORD *v8; // r15
  signed __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  char v14; // al
  _QWORD *v15; // r15
  __int64 v16; // r14
  unsigned __int64 v17; // rsi
  char v18; // dl
  __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // r15
  __int64 v22; // r14
  char v23; // r14
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // r15d
  __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rax
  __int64 v32; // r14
  unsigned __int8 v33; // dl
  unsigned __int64 v34; // rax
  unsigned int v35; // r15d
  unsigned __int64 v36; // r14
  char v37; // cl
  int v38; // ecx
  __int64 v39; // r8
  int v40; // esi
  unsigned int v41; // edx
  __int64 v42; // rax
  __int64 v43; // r9
  __int64 v44; // rbx
  __int64 v45; // rax
  unsigned int v46; // ecx
  __int64 v47; // r8
  int v48; // eax
  bool v49; // al
  _QWORD *v50; // rbx
  __int64 v51; // rax
  unsigned __int64 v52; // rdx
  __int64 v53; // rax
  unsigned int v54; // esi
  unsigned int v55; // eax
  __int64 v56; // rdi
  int v57; // edx
  __int64 v58; // rcx
  int v59; // eax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // rax
  __int64 v64; // rax
  __int64 v65; // rdx
  int v66; // r11d
  __int64 v67; // rcx
  int v68; // edx
  __int64 v69; // rax
  __int64 v70; // rsi
  __int64 v71; // rsi
  __int64 v72; // rcx
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rbx
  __int64 v76; // rcx
  __int64 v77; // rax
  __int64 v78; // r12
  __int64 v79; // rdi
  int v80; // eax
  int v81; // eax
  __int64 v82; // rsi
  int v83; // edx
  __int64 v84; // rax
  __int64 v85; // rcx
  int v86; // r10d
  __int64 v87; // r8
  int v88; // edx
  int v89; // edx
  int v90; // r10d
  unsigned __int64 v91; // [rsp+8h] [rbp-C8h]
  unsigned int v93; // [rsp+1Ch] [rbp-B4h]
  unsigned int v94; // [rsp+20h] [rbp-B0h]
  char v95; // [rsp+28h] [rbp-A8h]
  __int64 v96; // [rsp+28h] [rbp-A8h]
  __int64 v97; // [rsp+28h] [rbp-A8h]
  int v98; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v99; // [rsp+30h] [rbp-A0h]
  __int64 v100; // [rsp+38h] [rbp-98h]
  __int64 v101; // [rsp+38h] [rbp-98h]
  __int64 v102; // [rsp+40h] [rbp-90h]
  _QWORD *v104; // [rsp+50h] [rbp-80h] BYREF
  __int64 v105; // [rsp+58h] [rbp-78h]
  unsigned __int64 v106; // [rsp+60h] [rbp-70h] BYREF
  __int64 v107; // [rsp+68h] [rbp-68h]
  __int64 v108; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v109; // [rsp+78h] [rbp-58h]
  __int64 v110; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v111; // [rsp+88h] [rbp-48h]
  unsigned __int64 v112; // [rsp+90h] [rbp-40h] BYREF
  __int64 v113; // [rsp+98h] [rbp-38h]

  v102 = a1;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v7 = *(_QWORD *)(a1 - 8);
  else
    v7 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v8 = (_QWORD *)(v7 + 32);
  v104 = v8;
  v9 = sub_BB5290(a1) & 0xFFFFFFFFFFFFFFF9LL | 4;
  v105 = v9;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v102 = *(_QWORD *)(a1 - 8) + 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( v8 == (_QWORD *)v102 )
    return 1;
  v10 = 0;
  if ( a3 )
    v10 = 0xFFFFFFFFFFFFFFFFLL >> -(char)a3;
  v99 = v10;
  v91 = v10 & 1;
  while ( 1 )
  {
    v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v9 )
      goto LABEL_39;
    v13 = (v9 >> 1) & 3;
    if ( v13 != 2 )
    {
      if ( v13 == 1 && v11 )
      {
        v12 = *(_QWORD *)(v11 + 24);
        goto LABEL_15;
      }
LABEL_39:
      v12 = sub_BCBAE0(v11, *v8);
      goto LABEL_15;
    }
    if ( !v11 )
      goto LABEL_39;
LABEL_15:
    v14 = sub_BCEA30(v12);
    v15 = v104;
    v16 = v105;
    v17 = 0;
    v18 = v14;
    v19 = *v104;
    if ( v105 )
    {
      v17 = v105 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v105 & 6) != 0 )
        v17 = 0;
    }
    if ( *(_BYTE *)v19 != 17 || *(_BYTE *)(*(_QWORD *)(v19 + 8) + 8LL) != 12 )
      break;
    v46 = *(_DWORD *)(v19 + 32);
    v47 = v19 + 24;
    if ( v46 <= 0x40 )
    {
      v49 = *(_QWORD *)(v19 + 24) == 0;
    }
    else
    {
      v93 = *(_DWORD *)(v19 + 32);
      v95 = v14;
      v48 = sub_C444A0(v19 + 24);
      v46 = v93;
      v47 = v19 + 24;
      v18 = v95;
      v49 = v93 == v48;
    }
    if ( !v49 )
    {
      if ( v18 )
        return 0;
      if ( v17 )
      {
        v50 = *(_QWORD **)(v19 + 24);
        if ( v46 > 0x40 )
          v50 = (_QWORD *)*v50;
        v51 = 16LL * (unsigned int)v50 + sub_AE4AC0(a2, v17) + 24;
        v52 = *(_QWORD *)v51;
        LOBYTE(v51) = *(_BYTE *)(v51 + 8);
        v106 = v52;
        LOBYTE(v107) = v51;
        v53 = sub_CA1930(&v106);
        v109 = a3;
        if ( a3 > 0x40 )
        {
          sub_C43690(&v108, v53, 0);
          sub_C44B10(&v112, &v108, a3);
          if ( v109 > 0x40 && v108 )
            j_j___libc_free_0_0(v108);
          v108 = v112;
          v111 = a3;
          v109 = v113;
          sub_C43690(&v110, 1, 0);
        }
        else
        {
          v108 = v53;
          sub_C44B10(&v112, &v108, a3);
          if ( v109 > 0x40 && v108 )
            j_j___libc_free_0_0(v108);
          v111 = a3;
          v108 = v112;
          v109 = v113;
          v110 = v91;
        }
      }
      else
      {
        v100 = v47;
        v106 = sub_9914A0((__int64)&v104, a2);
        v107 = v65;
        v96 = sub_CA1930(&v106);
        v109 = *(_DWORD *)(v19 + 32);
        if ( v109 > 0x40 )
          sub_C43780(&v108, v100);
        else
          v108 = *(_QWORD *)(v19 + 24);
        sub_C44B10(&v112, &v108, a3);
        if ( v109 > 0x40 && v108 )
          j_j___libc_free_0_0(v108);
        v111 = a3;
        v108 = v112;
        v109 = v113;
        if ( a3 > 0x40 )
          sub_C43690(&v110, v96, 0);
        else
          v110 = v99 & v96;
      }
      sub_C472A0(&v112, &v108, &v110);
      sub_C45EE0(a5, &v112);
      if ( (unsigned int)v113 > 0x40 && v112 )
        j_j___libc_free_0_0(v112);
      if ( v111 > 0x40 && v110 )
        j_j___libc_free_0_0(v110);
      if ( v109 > 0x40 )
      {
        v29 = v108;
        if ( v108 )
          goto LABEL_31;
      }
      goto LABEL_32;
    }
LABEL_33:
    v30 = v16 & 0xFFFFFFFFFFFFFFF8LL;
    v31 = v16 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v16 )
      goto LABEL_40;
    v32 = (v16 >> 1) & 3;
    if ( v32 == 2 )
    {
      if ( v30 )
        goto LABEL_36;
LABEL_40:
      v31 = sub_BCBAE0(v30, *v15);
      v15 = v104;
      goto LABEL_36;
    }
    if ( v32 != 1 || !v30 )
      goto LABEL_40;
    v31 = *(_QWORD *)(v30 + 24);
LABEL_36:
    v33 = *(_BYTE *)(v31 + 8);
    if ( v33 == 16 )
    {
      v105 = *(_QWORD *)(v31 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
    }
    else
    {
      v34 = v31 & 0xFFFFFFFFFFFFFFF9LL;
      if ( (unsigned int)v33 - 17 > 1 )
      {
        if ( v33 != 15 )
          v34 = 0;
        v105 = v34;
      }
      else
      {
        v105 = v34 | 2;
      }
    }
    v8 = v15 + 4;
    v104 = v8;
    if ( (_QWORD *)v102 == v8 )
      return 1;
    v9 = v105;
  }
  if ( !v17 && !v14 )
  {
    v20 = v105 & 0xFFFFFFFFFFFFFFF8LL;
    v21 = v105 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v105 )
      goto LABEL_56;
    v22 = (v105 >> 1) & 3;
    if ( v22 == 2 )
    {
      if ( v20 )
        goto LABEL_25;
LABEL_56:
      v21 = sub_BCBAE0(v20, *v104);
      if ( ((v105 >> 1) & 3) == 1 )
        goto LABEL_57;
LABEL_25:
      v23 = sub_AE5020(a2, v21);
      v24 = sub_9208B0(a2, v21);
      v113 = v25;
      v26 = ((1LL << v23) + ((unsigned __int64)(v24 + 7) >> 3) - 1) >> v23 << v23;
    }
    else
    {
      if ( v22 != 1 || !v20 )
        goto LABEL_56;
      v21 = *(_QWORD *)(v20 + 24);
LABEL_57:
      v45 = sub_9208B0(a2, v21);
      v113 = v25;
      v26 = (unsigned __int64)(v45 + 7) >> 3;
    }
    v112 = v26;
    LOBYTE(v113) = v25;
    v27 = sub_CA1930(&v112);
    v111 = a3;
    if ( a3 <= 0x40 )
    {
      v28 = a3;
      v110 = v99 & v27;
      goto LABEL_28;
    }
    sub_C43690(&v110, v27, 0);
    v28 = v111;
    if ( v111 > 0x40 )
    {
      if ( (unsigned int)sub_C444A0(&v110) != v28 )
      {
        LODWORD(v113) = a3;
        goto LABEL_44;
      }
LABEL_29:
      if ( v28 > 0x40 )
      {
        v29 = v110;
        if ( v110 )
LABEL_31:
          j_j___libc_free_0_0(v29);
      }
LABEL_32:
      v16 = v105;
      v15 = v104;
      goto LABEL_33;
    }
LABEL_28:
    if ( !v110 )
      goto LABEL_29;
    LODWORD(v113) = a3;
    if ( a3 > 0x40 )
    {
LABEL_44:
      sub_C43690(&v112, 0, 0);
      v35 = v113;
      v36 = v112;
    }
    else
    {
      v112 = 0;
      v35 = a3;
      v36 = 0;
    }
    v37 = *(_BYTE *)(a4 + 8);
    LODWORD(v113) = 0;
    v38 = v37 & 1;
    if ( v38 )
    {
      v39 = a4 + 16;
      v40 = 3;
      goto LABEL_47;
    }
    v54 = *(_DWORD *)(a4 + 24);
    v39 = *(_QWORD *)(a4 + 16);
    if ( v54 )
    {
      v40 = v54 - 1;
LABEL_47:
      v41 = v40 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v42 = v39 + 16LL * v41;
      v43 = *(_QWORD *)v42;
      if ( v19 == *(_QWORD *)v42 )
      {
LABEL_48:
        v44 = *(_QWORD *)(a4 + 80) + 24LL * *(unsigned int *)(v42 + 8);
        goto LABEL_49;
      }
      v66 = 1;
      v56 = 0;
      while ( v43 != -4096 )
      {
        if ( !v56 && v43 == -8192 )
          v56 = v42;
        v41 = v40 & (v66 + v41);
        v42 = v39 + 16LL * v41;
        v43 = *(_QWORD *)v42;
        if ( v19 == *(_QWORD *)v42 )
          goto LABEL_48;
        ++v66;
      }
      if ( !v56 )
        v56 = v42;
      v55 = *(_DWORD *)(a4 + 8);
      ++*(_QWORD *)a4;
      v57 = (v55 >> 1) + 1;
      if ( (_BYTE)v38 )
      {
        v54 = 4;
        if ( (unsigned int)(4 * v57) >= 0xC )
        {
LABEL_126:
          sub_BB64D0(a4, 2 * v54);
          if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
          {
            v67 = a4 + 16;
            v68 = 3;
          }
          else
          {
            v88 = *(_DWORD *)(a4 + 24);
            v67 = *(_QWORD *)(a4 + 16);
            if ( !v88 )
              goto LABEL_178;
            v68 = v88 - 1;
          }
          LODWORD(v69) = v68 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v56 = v67 + 16LL * (unsigned int)v69;
          v70 = *(_QWORD *)v56;
          if ( v19 == *(_QWORD *)v56 )
            goto LABEL_129;
          v90 = 1;
          v87 = 0;
          while ( v70 != -4096 )
          {
            if ( !v87 && v70 == -8192 )
              v87 = v56;
            v69 = v68 & (unsigned int)(v69 + v90);
            v56 = v67 + 16 * v69;
            v70 = *(_QWORD *)v56;
            if ( v19 == *(_QWORD *)v56 )
              goto LABEL_129;
            ++v90;
          }
LABEL_153:
          if ( v87 )
            v56 = v87;
          goto LABEL_129;
        }
LABEL_96:
        if ( v54 - *(_DWORD *)(a4 + 12) - v57 <= v54 >> 3 )
        {
          sub_BB64D0(a4, v54);
          if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
          {
            v82 = a4 + 16;
            v83 = 3;
          }
          else
          {
            v89 = *(_DWORD *)(a4 + 24);
            v82 = *(_QWORD *)(a4 + 16);
            if ( !v89 )
            {
LABEL_178:
              *(_DWORD *)(a4 + 8) = (2 * (*(_DWORD *)(a4 + 8) >> 1) + 2) | *(_DWORD *)(a4 + 8) & 1;
              BUG();
            }
            v83 = v89 - 1;
          }
          LODWORD(v84) = v83 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v56 = v82 + 16LL * (unsigned int)v84;
          v85 = *(_QWORD *)v56;
          if ( v19 != *(_QWORD *)v56 )
          {
            v86 = 1;
            v87 = 0;
            while ( v85 != -4096 )
            {
              if ( !v87 && v85 == -8192 )
                v87 = v56;
              v84 = v83 & (unsigned int)(v84 + v86);
              v56 = v82 + 16 * v84;
              v85 = *(_QWORD *)v56;
              if ( v19 == *(_QWORD *)v56 )
                goto LABEL_129;
              ++v86;
            }
            goto LABEL_153;
          }
LABEL_129:
          v55 = *(_DWORD *)(a4 + 8);
        }
        *(_DWORD *)(a4 + 8) = (2 * (v55 >> 1) + 2) | v55 & 1;
        if ( *(_QWORD *)v56 != -4096 )
          --*(_DWORD *)(a4 + 12);
        *(_DWORD *)(v56 + 8) = 0;
        *(_QWORD *)v56 = v19;
        *(_DWORD *)(v56 + 8) = *(_DWORD *)(a4 + 88);
        v58 = *(unsigned int *)(a4 + 88);
        v59 = v58;
        if ( *(_DWORD *)(a4 + 92) <= (unsigned int)v58 )
        {
          v71 = a4 + 96;
          v60 = sub_C8D7D0(a4 + 80, a4 + 96, 0, 24, &v108);
          v72 = 24LL * *(unsigned int *)(a4 + 88);
          v73 = v72 + v60;
          if ( v72 + v60 )
          {
            *(_DWORD *)(v73 + 16) = v35;
            v35 = 0;
            *(_QWORD *)v73 = v19;
            *(_QWORD *)(v73 + 8) = v36;
            v72 = 24LL * *(unsigned int *)(a4 + 88);
          }
          v74 = *(_QWORD *)(a4 + 80);
          v75 = v74 + v72;
          if ( v74 != v74 + v72 )
          {
            v76 = v60;
            do
            {
              if ( v76 )
              {
                *(_QWORD *)v76 = *(_QWORD *)v74;
                *(_DWORD *)(v76 + 16) = *(_DWORD *)(v74 + 16);
                v71 = *(_QWORD *)(v74 + 8);
                *(_QWORD *)(v76 + 8) = v71;
                *(_DWORD *)(v74 + 16) = 0;
              }
              v74 += 24;
              v76 += 24;
            }
            while ( v75 != v74 );
            v77 = *(_QWORD *)(a4 + 80);
            v75 = v77 + 24LL * *(unsigned int *)(a4 + 88);
            if ( v75 != v77 )
            {
              v97 = v60;
              v94 = a3;
              v78 = *(_QWORD *)(a4 + 80);
              do
              {
                v75 -= 24;
                if ( *(_DWORD *)(v75 + 16) > 0x40u )
                {
                  v79 = *(_QWORD *)(v75 + 8);
                  if ( v79 )
                    j_j___libc_free_0_0(v79);
                }
              }
              while ( v75 != v78 );
              v60 = v97;
              a3 = v94;
              v75 = *(_QWORD *)(a4 + 80);
            }
          }
          v80 = v108;
          if ( v75 != a4 + 96 )
          {
            v98 = v108;
            v101 = v60;
            _libc_free(v75, v71);
            v80 = v98;
            v60 = v101;
          }
          *(_DWORD *)(a4 + 92) = v80;
          v81 = *(_DWORD *)(a4 + 88);
          *(_QWORD *)(a4 + 80) = v60;
          v64 = (unsigned int)(v81 + 1);
          *(_DWORD *)(a4 + 88) = v64;
        }
        else
        {
          v60 = *(_QWORD *)(a4 + 80);
          v61 = v60 + 24 * v58;
          if ( v61 )
          {
            *(_QWORD *)v61 = v19;
            *(_DWORD *)(v61 + 16) = v35;
            *(_QWORD *)(v61 + 8) = v36;
            v62 = (unsigned int)(*(_DWORD *)(a4 + 88) + 1);
            *(_DWORD *)(a4 + 88) = v62;
            v44 = *(_QWORD *)(a4 + 80) + 24 * v62 - 24;
            goto LABEL_52;
          }
          v64 = (unsigned int)(v59 + 1);
          *(_DWORD *)(a4 + 88) = v64;
        }
        v44 = v60 + 24 * v64 - 24;
LABEL_49:
        if ( v35 > 0x40 && v36 )
          j_j___libc_free_0_0(v36);
LABEL_52:
        if ( (unsigned int)v113 > 0x40 && v112 )
          j_j___libc_free_0_0(v112);
        sub_C45EE0(v44 + 8, &v110);
        v28 = v111;
        goto LABEL_29;
      }
      v54 = *(_DWORD *)(a4 + 24);
    }
    else
    {
      v55 = *(_DWORD *)(a4 + 8);
      ++*(_QWORD *)a4;
      v56 = 0;
      v57 = (v55 >> 1) + 1;
    }
    if ( 3 * v54 <= 4 * v57 )
      goto LABEL_126;
    goto LABEL_96;
  }
  return 0;
}
