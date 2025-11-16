// Function: sub_26847B0
// Address: 0x26847b0
//
__int64 __fastcall sub_26847B0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  unsigned int v5; // esi
  __int64 v6; // r8
  int v7; // r12d
  unsigned int v8; // edi
  __int64 *v9; // rdx
  __int64 *v10; // rax
  __int64 v11; // rcx
  __int64 *v12; // rbx
  __int64 v13; // r12
  int v15; // ecx
  int v16; // ecx
  unsigned __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  bool v20; // zf
  __int64 v21; // r12
  signed __int64 v22; // rbx
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 *v25; // r11
  __int64 v26; // rdx
  int v27; // ebx
  __int64 *v28; // rdx
  unsigned __int8 *v29; // rdi
  int v30; // eax
  __int64 v31; // rsi
  __int64 v32; // rsi
  __int64 *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned int v36; // esi
  __int64 v37; // rdi
  int v38; // ebx
  unsigned int v39; // edx
  __int64 *v40; // rcx
  __int64 *v41; // rax
  __int64 v42; // r11
  __int64 *v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 *v46; // rdi
  int v47; // edx
  unsigned int v48; // eax
  __int64 *v49; // rsi
  __int64 v50; // r8
  __int64 v51; // r15
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 *v55; // rax
  __int64 v56; // rax
  __int64 v57; // r15
  int v58; // eax
  int v59; // esi
  __int64 v60; // r8
  unsigned int v61; // edx
  __int64 v62; // rdi
  int v63; // ebx
  __int64 *v64; // r11
  int v65; // eax
  int v66; // edx
  __int64 v67; // rdi
  __int64 *v68; // r8
  unsigned int v69; // ebx
  int v70; // r11d
  __int64 v71; // rsi
  int v72; // esi
  int v73; // r11d
  __int64 *v74; // rax
  __int64 *v75; // rdx
  __int64 *v76; // rcx
  int v77; // ecx
  int v78; // edx
  __int64 v79; // [rsp+0h] [rbp-E0h]
  __int64 v80; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v81; // [rsp+30h] [rbp-B0h] BYREF
  __int64 *v82; // [rsp+38h] [rbp-A8h]
  __int64 v83; // [rsp+40h] [rbp-A0h]
  int v84; // [rsp+48h] [rbp-98h]
  char v85; // [rsp+4Ch] [rbp-94h]
  char v86; // [rsp+50h] [rbp-90h] BYREF
  __int64 *v87; // [rsp+60h] [rbp-80h] BYREF
  __int64 v88; // [rsp+68h] [rbp-78h]
  _BYTE v89[112]; // [rsp+70h] [rbp-70h] BYREF

  v3 = a2;
  v4 = **(_QWORD **)(a1 + 72);
  if ( v4 && *(_DWORD *)(v4 + 40) )
  {
    v44 = *(_QWORD *)(v4 + 8);
    v45 = *(unsigned int *)(v4 + 24);
    v46 = (__int64 *)(v44 + 8 * v45);
    if ( !(_DWORD)v45 )
      return 0;
    v47 = v45 - 1;
    v48 = v47 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v49 = (__int64 *)(v44 + 8LL * v48);
    v50 = *v49;
    if ( v3 != *v49 )
    {
      v72 = 1;
      while ( v50 != -4096 )
      {
        v73 = v72 + 1;
        v48 = v47 & (v72 + v48);
        v49 = (__int64 *)(v44 + 8LL * v48);
        v50 = *v49;
        if ( v3 == *v49 )
          goto LABEL_56;
        v72 = v73;
      }
      return 0;
    }
LABEL_56:
    if ( v46 == v49 )
      return 0;
  }
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_81;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v9 = (__int64 *)(v6 + 24LL * v8);
  v10 = 0;
  v11 = *v9;
  if ( v3 != *v9 )
  {
    while ( v11 != -4096 )
    {
      if ( !v10 && v11 == -8192 )
        v10 = v9;
      v8 = (v5 - 1) & (v7 + v8);
      v9 = (__int64 *)(v6 + 24LL * v8);
      v11 = *v9;
      if ( v3 == *v9 )
        goto LABEL_5;
      ++v7;
    }
    v15 = *(_DWORD *)(a1 + 16);
    if ( !v10 )
      v10 = v9;
    ++*(_QWORD *)a1;
    v16 = v15 + 1;
    if ( 4 * v16 < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 20) - v16 > v5 >> 3 )
      {
LABEL_18:
        *(_DWORD *)(a1 + 16) = v16;
        if ( *v10 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *v10 = v3;
        v12 = v10 + 1;
        *(_OWORD *)(v10 + 1) = 0;
        goto LABEL_21;
      }
      sub_26845B0(a1, v5);
      v65 = *(_DWORD *)(a1 + 24);
      if ( v65 )
      {
        v66 = v65 - 1;
        v67 = *(_QWORD *)(a1 + 8);
        v68 = 0;
        v69 = (v65 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
        v70 = 1;
        v16 = *(_DWORD *)(a1 + 16) + 1;
        v10 = (__int64 *)(v67 + 24LL * v69);
        v71 = *v10;
        if ( v3 != *v10 )
        {
          while ( v71 != -4096 )
          {
            if ( !v68 && v71 == -8192 )
              v68 = v10;
            v69 = v66 & (v70 + v69);
            v10 = (__int64 *)(v67 + 24LL * v69);
            v71 = *v10;
            if ( v3 == *v10 )
              goto LABEL_18;
            ++v70;
          }
          if ( v68 )
            v10 = v68;
        }
        goto LABEL_18;
      }
LABEL_131:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
LABEL_81:
    sub_26845B0(a1, 2 * v5);
    v58 = *(_DWORD *)(a1 + 24);
    if ( v58 )
    {
      v59 = v58 - 1;
      v60 = *(_QWORD *)(a1 + 8);
      v61 = (v58 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v16 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (__int64 *)(v60 + 24LL * v61);
      v62 = *v10;
      if ( v3 != *v10 )
      {
        v63 = 1;
        v64 = 0;
        while ( v62 != -4096 )
        {
          if ( v62 == -8192 && !v64 )
            v64 = v10;
          v61 = v59 & (v63 + v61);
          v10 = (__int64 *)(v60 + 24LL * v61);
          v62 = *v10;
          if ( v3 == *v10 )
            goto LABEL_18;
          ++v63;
        }
        if ( v64 )
          v10 = v64;
      }
      goto LABEL_18;
    }
    goto LABEL_131;
  }
LABEL_5:
  v12 = v9 + 1;
  if ( *((_BYTE *)v9 + 16) )
    return v9[1];
LABEL_21:
  if ( (unsigned __int8)sub_26747F0(v3) )
  {
    *v12 = v3;
    v13 = v3;
    *((_BYTE *)v12 + 8) = 1;
    return v13;
  }
  v20 = *((_BYTE *)v12 + 8) == 0;
  *v12 = 0;
  if ( v20 )
    *((_BYTE *)v12 + 8) = 1;
  if ( (*(_BYTE *)(v3 + 32) & 0xFu) - 7 > 1 )
  {
    sub_267C810(a1, v3, "OMP100", 6u);
    return 0;
  }
  v21 = *(_QWORD *)(v3 + 16);
  v22 = 0;
  v81 = 0;
  v82 = (__int64 *)&v86;
  v87 = (__int64 *)v89;
  v88 = 0x800000000LL;
  v23 = v21;
  v83 = 2;
  v84 = 0;
  v85 = 1;
  if ( v21 )
  {
    do
    {
      v23 = *(_QWORD *)(v23 + 8);
      ++v22;
    }
    while ( v23 );
    v24 = (__int64 *)v89;
    if ( v22 > 8 )
    {
      sub_C8D5F0((__int64)&v87, v89, v22, 8u, v18, v19);
      v24 = &v87[(unsigned int)v88];
    }
    do
    {
      *v24 = v21;
      v21 = *(_QWORD *)(v21 + 8);
      ++v24;
    }
    while ( v21 );
    v25 = v87;
    LODWORD(v88) = v22 + v88;
    if ( (_DWORD)v88 )
    {
      v26 = 0;
      v27 = 0;
      do
      {
        v28 = (__int64 *)v25[v26];
        v29 = (unsigned __int8 *)v28[3];
        v30 = *v29;
        if ( (_BYTE)v30 == 5 )
        {
          v57 = *((_QWORD *)v29 + 2);
          v34 = (unsigned int)v88;
          if ( v57 )
          {
            do
            {
              v17 = HIDWORD(v88);
              if ( v34 + 1 > (unsigned __int64)HIDWORD(v88) )
              {
                sub_C8D5F0((__int64)&v87, v89, v34 + 1, 8u, v18, v19);
                v34 = (unsigned int)v88;
              }
              v87[v34] = v57;
              v34 = (unsigned int)(v88 + 1);
              LODWORD(v88) = v88 + 1;
              v57 = *(_QWORD *)(v57 + 8);
            }
            while ( v57 );
            v25 = v87;
          }
          goto LABEL_45;
        }
        if ( (_BYTE)v30 == 82 )
        {
          if ( (*((_WORD *)v29 + 1) & 0x3Fu) - 32 > 1 )
            goto LABEL_39;
        }
        else
        {
          if ( (unsigned __int8)v30 <= 0x1Cu )
            goto LABEL_39;
          v17 = (unsigned int)(v30 - 34);
          if ( (unsigned __int8)(v30 - 34) > 0x33u )
            goto LABEL_39;
          v31 = 0x8000000000041LL;
          if ( !_bittest64(&v31, v17) )
            goto LABEL_39;
          v17 = (unsigned __int64)(v29 - 32);
          if ( v28 != (__int64 *)(v29 - 32) )
          {
            if ( (_BYTE)v30 != 85 )
              goto LABEL_39;
            v51 = *(_QWORD *)(a1 + 72);
            if ( (v29[7] & 0x80u) != 0 )
            {
              v79 = v28[3];
              v52 = sub_BD2BC0((__int64)v29);
              v29 = (unsigned __int8 *)v79;
              v53 = (__int64)v28 + v52;
              if ( *(char *)(v79 + 7) >= 0 )
              {
                v54 = v53 >> 4;
              }
              else
              {
                v29 = (unsigned __int8 *)v79;
                v54 = (v53 - sub_BD2BC0(v79)) >> 4;
              }
              if ( (_DWORD)v54 )
                goto LABEL_39;
            }
            v28 = *(__int64 **)(v51 + 28912);
            if ( !v28
              || (v55 = (__int64 *)*((_QWORD *)v29 - 4)) == 0
              || *(_BYTE *)v55
              || (v17 = *((_QWORD *)v29 + 10), v55[3] != v17)
              || v28 != v55 )
            {
LABEL_39:
              v32 = 0;
              if ( !v85 )
                goto LABEL_69;
              goto LABEL_40;
            }
          }
        }
        v56 = sub_B43CB0((__int64)v29);
        v32 = sub_26847B0(a1, v56);
        if ( !v85 )
          goto LABEL_69;
LABEL_40:
        v33 = v82;
        v17 = HIDWORD(v83);
        v28 = &v82[HIDWORD(v83)];
        if ( v82 != v28 )
        {
          while ( v32 != *v33 )
          {
            if ( v28 == ++v33 )
              goto LABEL_70;
          }
          goto LABEL_44;
        }
LABEL_70:
        if ( HIDWORD(v83) < (unsigned int)v83 )
        {
          v17 = (unsigned int)++HIDWORD(v83);
          *v28 = v32;
          ++v81;
          goto LABEL_44;
        }
LABEL_69:
        sub_C8CC70((__int64)&v81, v32, (__int64)v28, v17, v18, v19);
LABEL_44:
        LODWORD(v34) = v88;
        v25 = v87;
LABEL_45:
        v26 = (unsigned int)(v27 + 1);
        v27 = v26;
      }
      while ( (unsigned int)v34 > (unsigned int)v26 );
    }
    if ( v25 != (__int64 *)v89 )
      _libc_free((unsigned __int64)v25);
  }
  v35 = HIDWORD(v83);
  v13 = 0;
  if ( HIDWORD(v83) - v84 == 1 )
  {
    v74 = v82;
    if ( !v85 )
      v35 = (unsigned int)v83;
    v75 = &v82[v35];
    v13 = *v82;
    if ( v82 != v75 )
    {
      while ( 1 )
      {
        v13 = *v74;
        v76 = v74;
        if ( (unsigned __int64)*v74 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v75 == ++v74 )
        {
          v13 = v76[1];
          break;
        }
      }
    }
  }
  v36 = *(_DWORD *)(a1 + 24);
  v80 = v3;
  if ( !v36 )
  {
    ++*(_QWORD *)a1;
    v87 = 0;
    goto LABEL_118;
  }
  v37 = *(_QWORD *)(a1 + 8);
  v38 = 1;
  v39 = (v36 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v40 = (__int64 *)(v37 + 24LL * v39);
  v41 = 0;
  v42 = *v40;
  if ( v3 != *v40 )
  {
    while ( v42 != -4096 )
    {
      if ( !v41 && v42 == -8192 )
        v41 = v40;
      v39 = (v36 - 1) & (v38 + v39);
      v40 = (__int64 *)(v37 + 24LL * v39);
      v42 = *v40;
      if ( v3 == *v40 )
        goto LABEL_51;
      ++v38;
    }
    if ( !v41 )
      v41 = v40;
    v77 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v78 = v77 + 1;
    v87 = v41;
    if ( 4 * (v77 + 1) < 3 * v36 )
    {
      if ( v36 - *(_DWORD *)(a1 + 20) - v78 > v36 >> 3 )
      {
LABEL_114:
        *(_DWORD *)(a1 + 16) = v78;
        if ( *v41 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *v41 = v3;
        v43 = v41 + 1;
        *(_OWORD *)v43 = 0;
        goto LABEL_52;
      }
LABEL_119:
      sub_26845B0(a1, v36);
      sub_2677EC0(a1, &v80, &v87);
      v3 = v80;
      v78 = *(_DWORD *)(a1 + 16) + 1;
      v41 = v87;
      goto LABEL_114;
    }
LABEL_118:
    v36 *= 2;
    goto LABEL_119;
  }
LABEL_51:
  v43 = v40 + 1;
LABEL_52:
  *v43 = v13;
  *((_BYTE *)v43 + 8) = 1;
  if ( !v85 )
    _libc_free((unsigned __int64)v82);
  return v13;
}
