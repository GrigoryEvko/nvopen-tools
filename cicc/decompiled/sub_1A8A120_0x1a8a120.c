// Function: sub_1A8A120
// Address: 0x1a8a120
//
__int64 __fastcall sub_1A8A120(_QWORD *a1)
{
  _QWORD *v1; // r15
  _QWORD *v2; // r11
  _QWORD *v3; // rax
  __int64 v4; // rdx
  _QWORD *v5; // r9
  __int64 v6; // rbx
  _QWORD *v7; // r15
  __int64 i; // r12
  __int64 v9; // rbx
  __int64 v10; // r13
  unsigned __int64 v11; // rbx
  _QWORD *v12; // r14
  _QWORD *v13; // r15
  unsigned int v14; // eax
  _QWORD *v15; // r13
  unsigned __int64 v16; // rbx
  __int64 v17; // rax
  _QWORD *v18; // rbx
  __int64 v19; // rcx
  __int64 v20; // rax
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 v23; // rdi
  unsigned int v24; // r12d
  _QWORD *v26; // r12
  _QWORD *v27; // r13
  _QWORD *v28; // rax
  unsigned int v29; // r8d
  unsigned int v30; // edx
  __int64 v31; // rcx
  _QWORD *v32; // r13
  int *v33; // rax
  int *v34; // r15
  int *v35; // rax
  int *v36; // rdx
  unsigned __int64 v37; // rcx
  _QWORD *v38; // r9
  _QWORD *v39; // r10
  _QWORD *v40; // rsi
  _BYTE *v41; // rdx
  __int64 v42; // r8
  _BYTE *v43; // rsi
  __int64 v44; // r10
  _QWORD *v45; // r9
  int v46; // r10d
  _QWORD *v47; // rax
  int v48; // edx
  unsigned int v49; // ecx
  __int64 v50; // r9
  int v51; // edi
  _QWORD *v52; // rsi
  unsigned int v53; // eax
  _QWORD *v54; // rbx
  _QWORD *v55; // r12
  __int64 v56; // rsi
  _QWORD *v57; // rcx
  unsigned int v58; // r14d
  int v59; // esi
  __int64 v60; // r8
  _QWORD *v61; // r8
  _BYTE *v62; // r11
  _BYTE *v63; // r11
  _BYTE *v64; // r9
  int v65; // r11d
  unsigned int v66; // r10d
  _BYTE *v67; // rdi
  _BYTE *v68; // rax
  _QWORD *v69; // r11
  _BYTE *v70; // rdi
  _BYTE *v71; // rax
  _QWORD *v72; // r9
  _QWORD *v73; // [rsp+8h] [rbp-118h]
  __int64 v74; // [rsp+18h] [rbp-108h]
  _QWORD *v75; // [rsp+20h] [rbp-100h]
  _QWORD *v77; // [rsp+30h] [rbp-F0h]
  _QWORD *v78; // [rsp+38h] [rbp-E8h]
  __int64 v79; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v80; // [rsp+48h] [rbp-D8h]
  __int64 v81; // [rsp+50h] [rbp-D0h]
  unsigned int v82; // [rsp+58h] [rbp-C8h]
  void *v83; // [rsp+60h] [rbp-C0h]
  _QWORD v84[2]; // [rsp+68h] [rbp-B8h] BYREF
  __int64 v85; // [rsp+78h] [rbp-A8h]
  __int64 v86; // [rsp+80h] [rbp-A0h]
  void *v87; // [rsp+90h] [rbp-90h] BYREF
  __int64 v88; // [rsp+98h] [rbp-88h] BYREF
  _QWORD *v89; // [rsp+A0h] [rbp-80h]
  __int64 v90; // [rsp+A8h] [rbp-78h]
  __int64 v91; // [rsp+B0h] [rbp-70h]
  __int64 v92; // [rsp+C0h] [rbp-60h] BYREF
  int v93; // [rsp+C8h] [rbp-58h] BYREF
  __int64 v94; // [rsp+D0h] [rbp-50h]
  int *v95; // [rsp+D8h] [rbp-48h]
  int *v96; // [rsp+E0h] [rbp-40h]
  __int64 v97; // [rsp+E8h] [rbp-38h]

  v1 = (_QWORD *)*a1;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v93 = 0;
  v94 = 0;
  v95 = &v93;
  v96 = &v93;
  v97 = 0;
  if ( v1 == a1 )
  {
    v23 = 0;
    v24 = 0;
    goto LABEL_47;
  }
  v2 = v1;
  do
  {
    while ( 1 )
    {
      v75 = v2 + 2;
      v3 = (_QWORD *)v2[4];
      v4 = v3 == (_QWORD *)v2[3] ? *((unsigned int *)v2 + 11) : *((unsigned int *)v2 + 10);
      v5 = &v3[v4];
      if ( v3 != v5 )
      {
        while ( 1 )
        {
          v6 = *v3;
          if ( *v3 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v5 == ++v3 )
            goto LABEL_8;
        }
        if ( v5 != v3 )
          break;
      }
LABEL_8:
      v2 = (_QWORD *)*v2;
      if ( a1 == v2 )
        goto LABEL_9;
    }
    v77 = v2;
    v26 = v3;
    v27 = v5;
    if ( *(_BYTE *)(v6 + 16) != 54 )
      goto LABEL_50;
LABEL_57:
    if ( !v82 )
    {
      ++v79;
      goto LABEL_103;
    }
    v29 = v82 - 1;
    v30 = (v82 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v31 = *(_QWORD *)(v80 + 16LL * v30);
    v74 = v80 + 16LL * v30;
    if ( v31 == v6 )
      goto LABEL_59;
    v45 = (_QWORD *)(v80 + 16LL * ((v82 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4))));
    v46 = 1;
    v47 = 0;
    while ( 1 )
    {
      if ( v31 == -8 )
      {
        if ( !v47 )
          v47 = v45;
        ++v79;
        v48 = v81 + 1;
        if ( 4 * ((int)v81 + 1) < 3 * v82 )
        {
          if ( v82 - HIDWORD(v81) - v48 > v82 >> 3 )
            goto LABEL_97;
          sub_1A89A70((__int64)&v79, v82);
          if ( v82 )
          {
            v57 = 0;
            v58 = (v82 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
            v48 = v81 + 1;
            v59 = 1;
            v47 = (_QWORD *)(v80 + 16LL * v58);
            v60 = *v47;
            if ( *v47 != v6 )
            {
              while ( v60 != -8 )
              {
                if ( !v57 && v60 == -16 )
                  v57 = v47;
                v58 = (v82 - 1) & (v59 + v58);
                v47 = (_QWORD *)(v80 + 16LL * v58);
                v60 = *v47;
                if ( *v47 == v6 )
                  goto LABEL_97;
                ++v59;
              }
              if ( v57 )
                v47 = v57;
            }
            goto LABEL_97;
          }
          goto LABEL_160;
        }
LABEL_103:
        sub_1A89A70((__int64)&v79, 2 * v82);
        if ( v82 )
        {
          v49 = (v82 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
          v48 = v81 + 1;
          v47 = (_QWORD *)(v80 + 16LL * v49);
          v50 = *v47;
          if ( *v47 != v6 )
          {
            v51 = 1;
            v52 = 0;
            while ( v50 != -8 )
            {
              if ( v50 == -16 && !v52 )
                v52 = v47;
              v49 = (v82 - 1) & (v51 + v49);
              v47 = (_QWORD *)(v80 + 16LL * v49);
              v50 = *v47;
              if ( *v47 == v6 )
                goto LABEL_97;
              ++v51;
            }
            if ( v52 )
              v47 = v52;
          }
LABEL_97:
          LODWORD(v81) = v48;
          if ( *v47 != -8 )
            --HIDWORD(v81);
          *v47 = v6;
          v47[1] = v75;
          goto LABEL_50;
        }
LABEL_160:
        LODWORD(v81) = v81 + 1;
        BUG();
      }
      if ( v31 == -16 && !v47 )
        v47 = v45;
      v65 = v46 + 1;
      v66 = v30 + v46;
      v30 = v29 & v66;
      v45 = (_QWORD *)(v80 + 16LL * (v29 & v66));
      v31 = *v45;
      if ( *v45 == v6 )
        break;
      v46 = v65;
    }
    v74 = v80 + 16LL * (v29 & v66);
LABEL_59:
    v73 = v27;
    v32 = v77;
    while ( 2 )
    {
      v32 = (_QWORD *)v32[1];
      v87 = &v87;
      v88 = 1;
      v89 = v75;
      v33 = (int *)sub_1A89360(&v92, (__int64)&v87);
      v89 = v32 + 2;
      v87 = &v87;
      v34 = v33;
      v88 = 1;
      v35 = (int *)sub_1A89360(&v92, (__int64)&v87);
      v36 = v35;
      if ( v35 == &v93 )
      {
        if ( v34 == &v93 )
          goto LABEL_69;
        if ( (v34[10] & 1) != 0 )
        {
          *(_QWORD *)(*((_QWORD *)v34 + 4) + 8LL) &= 1uLL;
          BUG();
        }
        v41 = (_BYTE *)*((_QWORD *)v34 + 4);
        v37 = 0;
        if ( (v41[8] & 1) == 0 )
        {
LABEL_74:
          v42 = *(_QWORD *)v41;
          if ( (*(_BYTE *)(*(_QWORD *)v41 + 8LL) & 1) != 0 )
          {
            v41 = *(_BYTE **)v41;
          }
          else
          {
            v43 = *(_BYTE **)v42;
            if ( (*(_BYTE *)(*(_QWORD *)v42 + 8LL) & 1) == 0 )
            {
              v44 = *(_QWORD *)v43;
              if ( (*(_BYTE *)(*(_QWORD *)v43 + 8LL) & 1) != 0 )
              {
                v43 = *(_BYTE **)v43;
              }
              else
              {
                v63 = *(_BYTE **)v44;
                if ( (*(_BYTE *)(*(_QWORD *)v44 + 8LL) & 1) == 0 )
                {
                  v64 = *(_BYTE **)v63;
                  if ( (*(_BYTE *)(*(_QWORD *)v63 + 8LL) & 1) == 0 )
                  {
                    v70 = *(_BYTE **)v64;
                    if ( (*(_BYTE *)(*(_QWORD *)v64 + 8LL) & 1) == 0 )
                    {
                      v71 = sub_1A89840(v70);
                      *v72 = v71;
                      v70 = v71;
                    }
                    *(_QWORD *)v63 = v70;
                    v64 = v70;
                  }
                  *(_QWORD *)v44 = v64;
                  v63 = v64;
                }
                *(_QWORD *)v43 = v63;
                v43 = v63;
              }
              *(_QWORD *)v42 = v43;
            }
            *(_QWORD *)v41 = v43;
            v41 = v43;
          }
          *((_QWORD *)v34 + 4) = v41;
          goto LABEL_84;
        }
LABEL_68:
        *(_QWORD *)(*(_QWORD *)v41 + 8LL) = v37 | *(_QWORD *)(*(_QWORD *)v41 + 8LL) & 1LL;
        *(_QWORD *)v41 = *(_QWORD *)v37;
        *(_QWORD *)(v37 + 8) &= ~1uLL;
        *(_QWORD *)v37 = v41;
        goto LABEL_69;
      }
      if ( (v35[10] & 1) != 0 )
      {
        v37 = (unsigned __int64)(v35 + 8);
LABEL_81:
        if ( v34 == &v93 )
          goto LABEL_159;
        goto LABEL_82;
      }
      v37 = *((_QWORD *)v35 + 4);
      if ( (*(_BYTE *)(v37 + 8) & 1) != 0 )
        goto LABEL_81;
      v38 = *(_QWORD **)v37;
      if ( (*(_BYTE *)(*(_QWORD *)v37 + 8LL) & 1) != 0 )
      {
        *((_QWORD *)v35 + 4) = v38;
        if ( v34 == &v93 )
LABEL_159:
          BUG();
        v37 = (unsigned __int64)v38;
        goto LABEL_82;
      }
      v39 = (_QWORD *)*v38;
      if ( (*(_BYTE *)(*v38 + 8LL) & 1) != 0 )
      {
        *(_QWORD *)v37 = v39;
        v37 = (unsigned __int64)v39;
        *((_QWORD *)v35 + 4) = v39;
        if ( v34 != &v93 )
          goto LABEL_82;
        v41 = 0;
        goto LABEL_68;
      }
      v40 = (_QWORD *)*v39;
      if ( (*(_BYTE *)(*v39 + 8LL) & 1) == 0 )
      {
        v61 = (_QWORD *)*v40;
        if ( (*(_BYTE *)(*v40 + 8LL) & 1) == 0 )
        {
          v62 = (_BYTE *)*v61;
          if ( (*(_BYTE *)(*v61 + 8LL) & 1) == 0 )
          {
            v67 = *(_BYTE **)v62;
            if ( (*(_BYTE *)(*(_QWORD *)v62 + 8LL) & 1) == 0 )
            {
              v68 = sub_1A89840(v67);
              *v69 = v68;
              v67 = v68;
            }
            *v61 = v67;
            v62 = v67;
          }
          *v40 = v62;
          v61 = v62;
        }
        *v39 = v61;
        *v38 = v61;
        *(_QWORD *)v37 = v61;
        v37 = (unsigned __int64)v61;
        *((_QWORD *)v36 + 4) = v61;
        v41 = 0;
        if ( v34 != &v93 )
          goto LABEL_82;
        goto LABEL_84;
      }
      *v38 = v40;
      *(_QWORD *)v37 = v40;
      v37 = (unsigned __int64)v40;
      *((_QWORD *)v35 + 4) = v40;
      if ( v34 == &v93 )
      {
        v41 = 0;
        goto LABEL_68;
      }
LABEL_82:
      v41 = v34 + 8;
      if ( (v34[10] & 1) == 0 )
      {
        v41 = (_BYTE *)*((_QWORD *)v34 + 4);
        if ( (v41[8] & 1) == 0 )
          goto LABEL_74;
      }
LABEL_84:
      if ( (_BYTE *)v37 != v41 )
        goto LABEL_68;
LABEL_69:
      if ( *(_QWORD **)(v74 + 8) != v32 + 2 )
        continue;
      break;
    }
    v27 = v73;
LABEL_50:
    while ( 1 )
    {
      v28 = v26 + 1;
      if ( v26 + 1 == v27 )
        break;
      while ( 1 )
      {
        v6 = *v28;
        v26 = v28;
        if ( *v28 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v27 == ++v28 )
          goto LABEL_53;
      }
      if ( v28 == v27 )
        break;
      if ( *(_BYTE *)(v6 + 16) == 54 )
        goto LABEL_57;
    }
LABEL_53:
    v2 = (_QWORD *)*v77;
  }
  while ( a1 != (_QWORD *)*v77 );
LABEL_9:
  v7 = v2;
  if ( v97 )
  {
    for ( i = (__int64)v95; (int *)i != &v93; i = sub_220EF30(i) )
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(i + 40);
        if ( (v9 & 1) != 0 )
        {
          v10 = *(_QWORD *)(i + 48);
          v11 = v9 & 0xFFFFFFFFFFFFFFFELL;
          if ( v11 )
            break;
        }
        i = sub_220EF30(i);
        if ( (int *)i == &v93 )
          goto LABEL_17;
      }
      do
      {
        sub_1A898D0(*(_QWORD *)(v11 + 16), v10);
        v11 = *(_QWORD *)(v11 + 8) & 0xFFFFFFFFFFFFFFFELL;
      }
      while ( v11 );
    }
LABEL_17:
    if ( v7 != (_QWORD *)*a1 )
    {
      v78 = v7;
      v12 = (_QWORD *)*a1;
      do
      {
        while ( 1 )
        {
          v13 = v12;
          v12 = (_QWORD *)*v12;
          if ( *((_DWORD *)v13 + 11) == *((_DWORD *)v13 + 12) )
            break;
          if ( v78 == v12 )
            goto LABEL_46;
        }
        --a1[2];
        sub_2208CA0(v13);
        if ( *((_BYTE *)v13 + 288) )
        {
          v53 = *((_DWORD *)v13 + 70);
          if ( v53 )
          {
            v54 = (_QWORD *)v13[33];
            v55 = &v54[2 * v53];
            do
            {
              if ( *v54 != -8 && *v54 != -4 )
              {
                v56 = v54[1];
                if ( v56 )
                  sub_161E7C0((__int64)(v54 + 1), v56);
              }
              v54 += 2;
            }
            while ( v55 != v54 );
          }
          j___libc_free_0(v13[33]);
        }
        v14 = *((_DWORD *)v13 + 62);
        if ( v14 )
        {
          v15 = (_QWORD *)v13[29];
          v84[0] = 2;
          v16 = (unsigned __int64)v14 << 6;
          v84[1] = 0;
          v17 = -8;
          v18 = (_QWORD *)((char *)v15 + v16);
          v85 = -8;
          v83 = &unk_49E6B50;
          v86 = 0;
          v88 = 2;
          v89 = 0;
          v90 = -16;
          v87 = &unk_49E6B50;
          v91 = 0;
          while ( 1 )
          {
            v19 = v15[3];
            if ( v17 != v19 )
            {
              v17 = v90;
              if ( v19 != v90 )
              {
                v20 = v15[7];
                if ( v20 != 0 && v20 != -8 && v20 != -16 )
                {
                  sub_1649B30(v15 + 5);
                  v19 = v15[3];
                }
                v17 = v19;
              }
            }
            *v15 = &unk_49EE2B0;
            if ( v17 != 0 && v17 != -8 && v17 != -16 )
              sub_1649B30(v15 + 1);
            v15 += 8;
            if ( v18 == v15 )
              break;
            v17 = v85;
          }
          v87 = &unk_49EE2B0;
          if ( v90 != 0 && v90 != -8 && v90 != -16 )
            sub_1649B30(&v88);
          v83 = &unk_49EE2B0;
          if ( v85 != 0 && v85 != -8 && v85 != -16 )
            sub_1649B30(v84);
        }
        j___libc_free_0(v13[29]);
        v21 = v13[18];
        if ( (_QWORD *)v21 != v13 + 20 )
          _libc_free(v21);
        v22 = v13[4];
        if ( v22 != v13[3] )
          _libc_free(v22);
        j_j___libc_free_0(v13, 304);
      }
      while ( v78 != v12 );
    }
LABEL_46:
    v23 = v94;
    v24 = 1;
  }
  else
  {
    v23 = v94;
    v24 = 0;
  }
LABEL_47:
  sub_1A89670(v23);
  j___libc_free_0(v80);
  return v24;
}
