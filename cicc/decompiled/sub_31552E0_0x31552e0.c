// Function: sub_31552E0
// Address: 0x31552e0
//
__int64 __fastcall sub_31552E0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r13
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // rcx
  char v9; // al
  char v10; // dl
  __int64 v11; // r8
  __int64 v12; // r9
  int v13; // eax
  char v14; // al
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r8
  __int64 v19; // r9
  char v20; // al
  char v21; // dl
  const char **v22; // rax
  const char **v23; // rbx
  int v25; // eax
  const char *v26; // rax
  const char **v27; // rsi
  __int64 *v28; // rdx
  _QWORD *v29; // rax
  char v30; // dl
  char v31; // bl
  unsigned __int64 v32; // rax
  const char *v33; // rax
  const char *v34; // rax
  _QWORD *v35; // rax
  __int64 *v36; // rdx
  __int64 v37; // rcx
  char v38; // al
  _QWORD *v39; // rax
  _QWORD *v40; // rax
  __int64 v41; // rdx
  int v42; // ecx
  char *v43; // rdi
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  int v46; // [rsp+28h] [rbp-3C8h]
  unsigned __int8 v47; // [rsp+2Fh] [rbp-3C1h]
  unsigned __int64 v48; // [rsp+30h] [rbp-3C0h]
  char v49; // [rsp+38h] [rbp-3B8h]
  __int64 v50; // [rsp+40h] [rbp-3B0h] BYREF
  unsigned __int64 v51; // [rsp+48h] [rbp-3A8h]
  char *v52; // [rsp+50h] [rbp-3A0h] BYREF
  int v53; // [rsp+58h] [rbp-398h]
  int v54; // [rsp+5Ch] [rbp-394h]
  _BYTE v55[16]; // [rsp+60h] [rbp-390h] BYREF
  char *v56[20]; // [rsp+70h] [rbp-380h] BYREF
  _QWORD *v57; // [rsp+110h] [rbp-2E0h] BYREF
  __int64 *v58; // [rsp+118h] [rbp-2D8h]
  const char *v59; // [rsp+120h] [rbp-2D0h]
  char *v60; // [rsp+128h] [rbp-2C8h] BYREF
  __int64 v61; // [rsp+130h] [rbp-2C0h]
  _BYTE v62[128]; // [rsp+138h] [rbp-2B8h] BYREF
  __int64 v63; // [rsp+1B8h] [rbp-238h] BYREF
  int v64; // [rsp+1C0h] [rbp-230h] BYREF
  _QWORD *v65; // [rsp+1C8h] [rbp-228h]
  int *v66; // [rsp+1D0h] [rbp-220h]
  int *v67; // [rsp+1D8h] [rbp-218h]
  __int64 v68; // [rsp+1E0h] [rbp-210h]
  const char *v69; // [rsp+1F0h] [rbp-200h] BYREF
  __int64 v70; // [rsp+1F8h] [rbp-1F8h] BYREF
  __int64 *v71; // [rsp+200h] [rbp-1F0h]
  const char *v72; // [rsp+208h] [rbp-1E8h]
  unsigned __int64 v73[2]; // [rsp+210h] [rbp-1E0h] BYREF
  _BYTE v74[136]; // [rsp+220h] [rbp-1D0h] BYREF
  int v75; // [rsp+2A8h] [rbp-148h] BYREF
  _QWORD *v76; // [rsp+2B0h] [rbp-140h]
  int *v77; // [rsp+2B8h] [rbp-138h]
  int *v78; // [rsp+2C0h] [rbp-130h]
  __int64 v79; // [rsp+2C8h] [rbp-128h]
  unsigned __int64 v80; // [rsp+2D0h] [rbp-120h] BYREF
  _QWORD *v81; // [rsp+2D8h] [rbp-118h] BYREF
  __int64 *v82; // [rsp+2E0h] [rbp-110h]
  const char *v83; // [rsp+2E8h] [rbp-108h]
  char *v84; // [rsp+2F0h] [rbp-100h] BYREF
  __int64 v85; // [rsp+2F8h] [rbp-F8h]
  _BYTE v86[136]; // [rsp+300h] [rbp-F0h] BYREF
  int v87; // [rsp+388h] [rbp-68h] BYREF
  _QWORD *v88; // [rsp+390h] [rbp-60h]
  int *v89; // [rsp+398h] [rbp-58h]
  int *v90; // [rsp+3A0h] [rbp-50h]
  __int64 v91; // [rsp+3A8h] [rbp-48h]
  char v92; // [rsp+3B0h] [rbp-40h]

  v3 = a1;
  sub_3154320((__int64 *)&v80, a2, a3);
  v7 = v80 & 0xFFFFFFFFFFFFFFFELL;
  v48 = v80 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v80 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_BYTE *)(a1 + 224) |= 3u;
    *(_QWORD *)a1 = v7;
    return v3;
  }
  v51 = 0;
  memset(v56, 0, 0x98u);
  v8 = 0;
  v54 = 1;
  v52 = v55;
  v46 = 0;
  v49 = 0;
  v47 = 0;
  do
  {
    v53 = 0;
    sub_3154960((__int64)&v57, a2, v6, v8);
    v9 = (unsigned __int8)v58 & 1;
    v10 = (2 * ((unsigned __int8)v58 & 1)) | (unsigned __int8)v58 & 0xFD;
    LOBYTE(v58) = v10;
    if ( v9 )
    {
      v3 = a1;
      LOBYTE(v58) = v10 & 0xFD;
      v32 = (unsigned __int64)v57;
      *(_BYTE *)(a1 + 224) |= 3u;
      v57 = 0;
      *(_QWORD *)a1 = v32 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_53;
    }
    if ( (_DWORD)v57 != 3 )
    {
      v3 = a1;
      v80 = (unsigned __int64)"Expected records before encountering more subcontexts";
      LOWORD(v84) = 259;
      sub_31542E0((__int64 *)&v69, a2, (void **)&v80);
      v33 = v69;
      *(_BYTE *)(a1 + 224) |= 3u;
      *(_QWORD *)a1 = (unsigned __int64)v33 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_51;
    }
    sub_A4B600((__int64)&v69, a2 + 16, 3, (__int64)&v52, 0);
    v13 = v70 & 1;
    v8 = (unsigned int)(2 * v13);
    v6 = (unsigned int)v8 | v70 & 0xFD;
    LOBYTE(v70) = (2 * v13) | v70 & 0xFD;
    if ( (_BYTE)v13 )
    {
      v3 = a1;
      LOBYTE(v70) = v6 & 0xFD;
      v34 = v69;
      *(_BYTE *)(a1 + 224) |= 3u;
      v69 = 0;
      *(_QWORD *)a1 = (unsigned __int64)v34 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_51;
    }
    if ( (_DWORD)v69 == 3 )
    {
      if ( a3 != 11 )
      {
        BYTE1(v84) = 1;
        v3 = a1;
        v26 = "The root context should not have a callee index";
        goto LABEL_49;
      }
      if ( v53 == 1 )
      {
        v8 = v47;
        v46 = *(_DWORD *)v52;
        if ( !v47 )
          v8 = 1;
        v47 = v8;
        goto LABEL_12;
      }
      BYTE1(v84) = 1;
      v3 = a1;
      v26 = "The callee index should have exactly one value";
LABEL_49:
      v80 = (unsigned __int64)v26;
      LOBYTE(v84) = 3;
      sub_31542E0(&v50, a2, (void **)&v80);
      *(_BYTE *)(v3 + 224) |= 3u;
      *(_QWORD *)v3 = v50 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v70 & 2) != 0 )
LABEL_40:
        sub_9CE230(&v69);
      if ( (v70 & 1) != 0 && v69 )
        (*(void (__fastcall **)(const char *))(*(_QWORD *)v69 + 8LL))(v69);
LABEL_51:
      if ( ((unsigned __int8)v58 & 2) == 0 )
      {
        if ( ((unsigned __int8)v58 & 1) != 0 && v57 )
          (*(void (__fastcall **)(_QWORD *))(*v57 + 8LL))(v57);
        goto LABEL_53;
      }
LABEL_35:
      sub_9CEF10(&v57);
    }
    if ( (_DWORD)v69 == 4 )
    {
      if ( LOBYTE(v56[18]) )
      {
        sub_3153680((__int64)v56, &v52, v6, v8, v11, v12);
        v25 = (int)v56[1];
      }
      else
      {
        v56[0] = (char *)&v56[2];
        v56[1] = (char *)0x1000000000LL;
        if ( !v53 )
        {
          LOBYTE(v56[18]) = 1;
          v3 = a1;
          goto LABEL_48;
        }
        sub_3153680((__int64)v56, &v52, v6, v8, v11, v12);
        LOBYTE(v56[18]) = 1;
        v25 = (int)v56[1];
      }
      if ( v25 )
      {
        LOBYTE(v6) = v70;
        if ( (v70 & 2) != 0 )
          goto LABEL_40;
LABEL_9:
        v6 &= 1u;
        if ( (_DWORD)v6 && v69 )
          (*(void (__fastcall **)(const char *))(*(_QWORD *)v69 + 8LL))(v69);
LABEL_12:
        v14 = (char)v58;
        if ( ((unsigned __int8)v58 & 2) != 0 )
          goto LABEL_35;
        goto LABEL_13;
      }
      v3 = a1;
LABEL_48:
      BYTE1(v84) = 1;
      v26 = "Empty counters. At least the entry counter (one value) was expected";
      goto LABEL_49;
    }
    if ( (_DWORD)v69 != 2 )
      goto LABEL_9;
    if ( v53 != 1 )
    {
      BYTE1(v84) = 1;
      v3 = a1;
      v26 = "The GUID record should have exactly one value";
      goto LABEL_49;
    }
    v49 = 1;
    v48 = *(_QWORD *)v52;
    v14 = (char)v58;
    if ( ((unsigned __int8)v58 & 2) != 0 )
      goto LABEL_35;
LABEL_13:
    if ( (v14 & 1) != 0 && v57 )
      (*(void (__fastcall **)(_QWORD *))(*v57 + 8LL))(v57);
  }
  while ( !v49 || !LOBYTE(v56[18]) || a3 == 11 && !v47 );
  v57 = 0;
  v58 = 0;
  v59 = (const char *)v48;
  v60 = v62;
  v61 = 0x1000000000LL;
  if ( LODWORD(v56[1]) )
    sub_3153680((__int64)&v60, v56, v6, v8, v11, v12);
  v64 = 0;
  v65 = 0;
  v66 = &v64;
  v67 = &v64;
  v68 = 0;
  while ( 1 )
  {
    if ( !sub_3154C60(a2, 11, v6, v8) )
    {
      v3 = a1;
      LODWORD(v51) = v46;
      v82 = v58;
      BYTE4(v51) = v47;
      v80 = v51;
      v35 = v57;
      v81 = v57;
      if ( v58 )
      {
        *v58 = (__int64)&v81;
        v35 = v57;
      }
      if ( v35 )
        v35[1] = &v81;
      v58 = 0;
      v57 = 0;
      v83 = v59;
      v84 = v86;
      v85 = 0x1000000000LL;
      if ( (_DWORD)v61 )
        sub_3153680((__int64)&v84, &v60, (unsigned int)v61, v15, v16, v17);
      if ( v65 )
      {
        v88 = v65;
        v87 = v64;
        v89 = v66;
        v90 = v67;
        v65[1] = &v87;
        v65 = 0;
        v91 = v68;
        v66 = &v64;
        v67 = &v64;
        v68 = 0;
      }
      else
      {
        v87 = 0;
        v88 = 0;
        v89 = &v87;
        v90 = &v87;
        v91 = 0;
      }
      v36 = v82;
      v37 = a1 + 8;
      v38 = *(_BYTE *)(a1 + 224) & 0xFC;
      *(_QWORD *)(a1 + 16) = v82;
      *(_BYTE *)(a1 + 224) = v38 | 2;
      *(_QWORD *)a1 = v80;
      v39 = v81;
      *(_QWORD *)(a1 + 8) = v81;
      if ( v36 )
      {
        *v36 = v37;
        v39 = v81;
      }
      if ( v39 )
        v39[1] = v37;
      v82 = 0;
      v81 = 0;
      *(_QWORD *)(a1 + 24) = v83;
      *(_QWORD *)(a1 + 32) = a1 + 48;
      *(_QWORD *)(a1 + 40) = 0x1000000000LL;
      if ( (_DWORD)v85 )
        sub_3153680(a1 + 32, &v84, (__int64)v36, v37, v16, v17);
      v40 = v88;
      v41 = a1 + 184;
      if ( v88 )
      {
        v42 = v87;
        *(_QWORD *)(a1 + 192) = v88;
        *(_DWORD *)(a1 + 184) = v42;
        *(_QWORD *)(a1 + 200) = v89;
        *(_QWORD *)(a1 + 208) = v90;
        v40[1] = v41;
        v88 = 0;
        *(_QWORD *)(a1 + 216) = v91;
        v89 = &v87;
        v90 = &v87;
        v91 = 0;
      }
      else
      {
        *(_DWORD *)(a1 + 184) = 0;
        *(_QWORD *)(a1 + 192) = 0;
        *(_QWORD *)(a1 + 200) = v41;
        *(_QWORD *)(a1 + 208) = v41;
        *(_QWORD *)(a1 + 216) = 0;
      }
      sub_31541A0(0);
      v43 = v84;
      if ( v84 == v86 )
        goto LABEL_114;
LABEL_113:
      _libc_free((unsigned __int64)v43);
LABEL_114:
      if ( v82 )
        *v82 = (__int64)v81;
      if ( v81 )
        v81[1] = v82;
      goto LABEL_118;
    }
    sub_31552E0(&v80, a2, 11);
    v20 = v92 & 1;
    v21 = (2 * (v92 & 1)) | v92 & 0xFD;
    v92 = v21;
    if ( v20 )
    {
      v3 = a1;
      v92 = v21 & 0xFD;
      v44 = v80;
      *(_BYTE *)(a1 + 224) |= 3u;
      v80 = 0;
      *(_QWORD *)a1 = v44 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_118;
    }
    v22 = (const char **)v65;
    if ( v65 )
    {
      v23 = (const char **)&v64;
      do
      {
        if ( *((_DWORD *)v22 + 8) < (unsigned int)v80 )
        {
          v22 = (const char **)v22[3];
        }
        else
        {
          v23 = v22;
          v22 = (const char **)v22[2];
        }
      }
      while ( v22 );
      if ( v23 != (const char **)&v64 && (unsigned int)v80 >= *((_DWORD *)v23 + 8) )
        goto LABEL_62;
    }
    else
    {
      v23 = (const char **)&v64;
    }
    v27 = v23;
    v69 = (const char *)&v80;
    v23 = (const char **)sub_3155210(&v63, (__int64)v23, (unsigned int **)&v69);
    if ( (v92 & 2) != 0 )
      goto LABEL_91;
LABEL_62:
    v28 = v82;
    v69 = v83;
    v29 = v81;
    v71 = v82;
    v70 = (__int64)v81;
    if ( v82 )
    {
      *v82 = (__int64)&v70;
      v29 = v81;
    }
    if ( v29 )
    {
      v28 = &v70;
      v29[1] = &v70;
    }
    v82 = 0;
    v81 = 0;
    v72 = v83;
    v73[0] = (unsigned __int64)v74;
    v73[1] = 0x1000000000LL;
    if ( (_DWORD)v85 )
      sub_3153680((__int64)v73, &v84, (__int64)v28, (unsigned int)v85, v18, v19);
    if ( v88 )
    {
      v76 = v88;
      v75 = v87;
      v77 = v89;
      v78 = v90;
      v88[1] = &v75;
      v88 = 0;
      v79 = v91;
      v89 = &v87;
      v90 = &v87;
      v91 = 0;
    }
    else
    {
      v75 = 0;
      v76 = 0;
      v77 = &v75;
      v78 = &v75;
      v79 = 0;
    }
    v27 = &v69;
    sub_3154710(v23 + 5, (__int64 *)&v69);
    v31 = v30;
    sub_31541A0(v76);
    if ( (_BYTE *)v73[0] != v74 )
      _libc_free(v73[0]);
    if ( v71 )
    {
      v6 = v70;
      *v71 = v70;
    }
    if ( v70 )
    {
      v6 = (__int64)v71;
      *(_QWORD *)(v70 + 8) = v71;
    }
    if ( !v31 )
      break;
    if ( (v92 & 2) != 0 )
      goto LABEL_91;
    if ( (v92 & 1) != 0 )
    {
      if ( v80 )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v80 + 8LL))(v80);
    }
    else
    {
      sub_31541A0(v88);
      if ( v84 != v86 )
        _libc_free((unsigned __int64)v84);
      if ( v82 )
      {
        v6 = (__int64)v81;
        *v82 = (__int64)v81;
      }
      if ( v81 )
      {
        v6 = (__int64)v82;
        v81[1] = v82;
      }
    }
  }
  v27 = (const char **)a2;
  v3 = a1;
  v69 = "Unexpected duplicate target (callee) at the same callsite.";
  LOWORD(v73[0]) = 259;
  sub_31542E0(&v50, a2, (void **)&v69);
  v45 = v50;
  *(_BYTE *)(a1 + 224) |= 3u;
  *(_QWORD *)a1 = v45 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v92 & 2) != 0 )
LABEL_91:
    sub_31551A0(&v80, (__int64)v27);
  if ( (v92 & 1) == 0 )
  {
    sub_31541A0(v88);
    v43 = v84;
    if ( v84 == v86 )
      goto LABEL_114;
    goto LABEL_113;
  }
  if ( v80 )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v80 + 8LL))(v80);
LABEL_118:
  sub_31541A0(v65);
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  if ( v58 )
    *v58 = (__int64)v57;
  if ( v57 )
    v57[1] = v58;
LABEL_53:
  if ( v52 != v55 )
    _libc_free((unsigned __int64)v52);
  if ( LOBYTE(v56[18]) && (char **)v56[0] != &v56[2] )
    _libc_free((unsigned __int64)v56[0]);
  return v3;
}
