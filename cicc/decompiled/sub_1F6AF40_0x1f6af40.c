// Function: sub_1F6AF40
// Address: 0x1f6af40
//
__int64 __fastcall sub_1F6AF40(__int64 a1, _QWORD *a2)
{
  _QWORD *v4; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // r13
  __int64 *v9; // rdi
  _QWORD *v10; // r14
  __int64 v11; // rax
  __int64 (*v12)(void); // rdx
  __int64 (*v13)(); // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // rsi
  __int64 v18; // r13
  _QWORD *v19; // rax
  unsigned int v20; // edx
  __int64 v21; // rax
  unsigned __int64 *v22; // r14
  __int64 v23; // rax
  _QWORD *v24; // rbx
  __int64 v25; // rax
  unsigned __int64 *v26; // r13
  unsigned int v27; // eax
  _QWORD *v28; // r14
  _QWORD *v29; // rbx
  __int64 v30; // r8
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 v33; // rdi
  __int64 *v34; // rdx
  __int64 v35; // rsi
  _QWORD *v36; // rax
  _QWORD *v37; // rbx
  _QWORD *v38; // r12
  unsigned __int64 v39; // r15
  __int16 v40; // ax
  __int64 v41; // rax
  bool v42; // cf
  char (__fastcall *v43)(__int64, __int64); // rax
  unsigned int v44; // r13d
  __int16 v45; // ax
  __int64 v46; // rdi
  __int64 *v47; // rdx
  __int64 v48; // rsi
  __int64 v49; // r13
  __int64 v50; // rcx
  __int64 v51; // rdx
  __int16 v52; // ax
  _QWORD *v53; // rax
  __int64 v54; // rbx
  __int64 v55; // rdx
  __int64 v56; // rax
  unsigned __int64 *v57; // r12
  unsigned int v58; // eax
  _QWORD *v59; // r13
  _QWORD *v60; // rbx
  __int64 v61; // r14
  __int64 v62; // rdi
  __int64 v63; // rax
  __int64 v64; // r14
  __int64 v65; // rax
  _QWORD *v66; // rdx
  __int64 v67; // r9
  __int64 v68; // rdi
  __int64 v69; // rax
  unsigned __int64 v70; // rbx
  __int64 v71; // rax
  _QWORD *v72; // rdx
  __int64 v73; // r9
  __int64 v74; // rdi
  _QWORD *v75; // [rsp+0h] [rbp-730h]
  _QWORD *v76; // [rsp+8h] [rbp-728h]
  __int64 v77; // [rsp+8h] [rbp-728h]
  _QWORD *v78; // [rsp+8h] [rbp-728h]
  __int64 v79; // [rsp+10h] [rbp-720h]
  _QWORD *v80; // [rsp+10h] [rbp-720h]
  _QWORD *v81; // [rsp+10h] [rbp-720h]
  unsigned __int8 v82; // [rsp+18h] [rbp-718h]
  __int64 v83; // [rsp+18h] [rbp-718h]
  __int64 v84; // [rsp+20h] [rbp-710h]
  _QWORD *v85; // [rsp+20h] [rbp-710h]
  _QWORD *i; // [rsp+28h] [rbp-708h]
  __int64 v87; // [rsp+30h] [rbp-700h] BYREF
  __int64 v88; // [rsp+38h] [rbp-6F8h] BYREF
  _QWORD v89[29]; // [rsp+40h] [rbp-6F0h] BYREF
  _QWORD v90[11]; // [rsp+128h] [rbp-608h] BYREF
  char v91; // [rsp+180h] [rbp-5B0h] BYREF
  _QWORD *v92; // [rsp+1A0h] [rbp-590h]
  __int64 v93; // [rsp+1A8h] [rbp-588h]
  _QWORD v94[4]; // [rsp+1B0h] [rbp-580h] BYREF
  unsigned __int64 v95[20]; // [rsp+1D0h] [rbp-560h] BYREF
  unsigned __int64 v96; // [rsp+270h] [rbp-4C0h]
  unsigned __int64 v97; // [rsp+288h] [rbp-4A8h]
  unsigned __int64 v98; // [rsp+2A0h] [rbp-490h]
  _BYTE *v99; // [rsp+2B8h] [rbp-478h]
  _BYTE v100[776]; // [rsp+2C8h] [rbp-468h] BYREF
  __int64 v101; // [rsp+5D0h] [rbp-160h]
  unsigned __int64 v102; // [rsp+5D8h] [rbp-158h]
  __int64 v103; // [rsp+6F0h] [rbp-40h]

  v4 = (_QWORD *)(*a2 + 112LL);
  v87 = sub_1560340(v4, -1, "function-instrument", 0x13u);
  if ( !sub_155D460(&v87, 0) && sub_155D3E0((__int64)&v87) )
  {
    v6 = sub_155D8B0(&v87);
    if ( v7 == 11 && *(_QWORD *)v6 == 0x776C612D79617278LL && *(_WORD *)(v6 + 8) == 31073 && *(_BYTE *)(v6 + 10) == 115 )
    {
      v88 = sub_1560340(v4, -1, "xray-instruction-threshold", 0x1Au);
      i = a2 + 40;
      goto LABEL_11;
    }
  }
  v88 = sub_1560340(v4, -1, "xray-instruction-threshold", 0x1Au);
  if ( sub_155D460(&v88, 0) )
    return 0;
  if ( !sub_155D3E0((__int64)&v88) )
    return 0;
  v15 = sub_155D8B0(&v88);
  if ( sub_16D2B80(v15, v16, 0xAu, v95) )
    return 0;
  v83 = LODWORD(v95[0]);
  if ( v95[0] != LODWORD(v95[0]) )
    return 0;
  v17 = (_QWORD *)a2[41];
  v18 = 0;
  for ( i = a2 + 40; i != v17; v17 = (_QWORD *)v17[1] )
  {
    v19 = (_QWORD *)v17[4];
    if ( v17 + 3 != v19 )
    {
      v20 = 0;
      do
      {
        v19 = (_QWORD *)v19[1];
        ++v20;
      }
      while ( v17 + 3 != v19 );
      v18 += v20;
    }
  }
  v21 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FC62EC, 1u);
  if ( !v21 )
  {
    sub_1E056B0((__int64)v95);
    goto LABEL_53;
  }
  v22 = (unsigned __int64 *)(*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v21 + 104LL))(v21, &unk_4FC62EC);
  sub_1E056B0((__int64)v95);
  if ( !v22 )
  {
LABEL_53:
    if ( !v103 )
    {
      v63 = sub_22077B0(80);
      if ( v63 )
      {
        *(_QWORD *)(v63 + 24) = 0;
        *(_QWORD *)v63 = v63 + 16;
        *(_QWORD *)(v63 + 8) = 0x100000000LL;
        *(_QWORD *)(v63 + 32) = 0;
        *(_QWORD *)(v63 + 40) = 0;
        *(_DWORD *)(v63 + 48) = 0;
        *(_QWORD *)(v63 + 64) = 0;
        *(_BYTE *)(v63 + 72) = 0;
        *(_DWORD *)(v63 + 76) = 0;
      }
      v64 = v103;
      v103 = v63;
      if ( v64 )
      {
        v65 = *(unsigned int *)(v64 + 48);
        if ( (_DWORD)v65 )
        {
          v66 = *(_QWORD **)(v64 + 32);
          v85 = &v66[2 * v65];
          do
          {
            if ( *v66 != -16 && *v66 != -8 )
            {
              v67 = v66[1];
              if ( v67 )
              {
                v68 = *(_QWORD *)(v67 + 24);
                if ( v68 )
                {
                  v76 = v66;
                  v79 = v66[1];
                  j_j___libc_free_0(v68, *(_QWORD *)(v67 + 40) - v68);
                  v66 = v76;
                  v67 = v79;
                }
                v80 = v66;
                j_j___libc_free_0(v67, 56);
                v66 = v80;
              }
            }
            v66 += 2;
          }
          while ( v85 != v66 );
        }
        j___libc_free_0(*(_QWORD *)(v64 + 32));
        if ( *(_QWORD *)v64 != v64 + 16 )
          _libc_free(*(_QWORD *)v64);
        j_j___libc_free_0(v64, 80);
      }
    }
    v22 = v95;
    sub_1E06620((__int64)v95);
    v32 = v103;
    *(_QWORD *)(v103 + 64) = a2;
    sub_1E07D70(v32, 0);
  }
  v23 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FC6A0C, 1u);
  v24 = (_QWORD *)v23;
  if ( v23 )
    v24 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v23 + 104LL))(v23, &unk_4FC6A0C);
  sub_1F6AA30((__int64)v89, (__int64)&unk_4FC6A0C);
  memset(v90, 0, 72);
  v89[0] = &unk_49FBD08;
  v90[9] = &v91;
  v90[10] = 0x400000000LL;
  v92 = v94;
  v93 = 0;
  v94[0] = 0;
  v94[1] = 1;
  v25 = sub_163A1D0();
  sub_1E29510(v25);
  if ( !v24 )
  {
    if ( !v22[164] )
    {
      v69 = sub_22077B0(80);
      if ( v69 )
      {
        *(_QWORD *)(v69 + 24) = 0;
        *(_QWORD *)v69 = v69 + 16;
        *(_QWORD *)(v69 + 8) = 0x100000000LL;
        *(_QWORD *)(v69 + 32) = 0;
        *(_QWORD *)(v69 + 40) = 0;
        *(_DWORD *)(v69 + 48) = 0;
        *(_QWORD *)(v69 + 64) = 0;
        *(_BYTE *)(v69 + 72) = 0;
        *(_DWORD *)(v69 + 76) = 0;
      }
      v70 = v22[164];
      v22[164] = v69;
      if ( v70 )
      {
        v71 = *(unsigned int *)(v70 + 48);
        if ( (_DWORD)v71 )
        {
          v72 = *(_QWORD **)(v70 + 32);
          v81 = &v72[2 * v71];
          do
          {
            if ( *v72 != -8 && *v72 != -16 )
            {
              v73 = v72[1];
              if ( v73 )
              {
                v74 = *(_QWORD *)(v73 + 24);
                if ( v74 )
                {
                  v75 = v72;
                  v77 = v72[1];
                  j_j___libc_free_0(v74, *(_QWORD *)(v73 + 40) - v74);
                  v72 = v75;
                  v73 = v77;
                }
                v78 = v72;
                j_j___libc_free_0(v73, 56);
                v72 = v78;
              }
            }
            v72 += 2;
          }
          while ( v81 != v72 );
        }
        j___libc_free_0(*(_QWORD *)(v70 + 32));
        if ( *(_QWORD *)v70 != v70 + 16 )
          _libc_free(*(_QWORD *)v70);
        j_j___libc_free_0(v70, 80);
      }
    }
    sub_1E06620((__int64)v22);
    sub_1E2B150((__int64)v90, v22[164]);
    v24 = v89;
  }
  if ( v24[34] == v24[33] && v83 > v18 )
  {
    sub_1E290B0((__int64)v89);
    v57 = (unsigned __int64 *)v103;
    v95[0] = (unsigned __int64)&unk_49FB698;
    if ( v103 )
    {
      v58 = *(_DWORD *)(v103 + 48);
      if ( v58 )
      {
        v59 = *(_QWORD **)(v103 + 32);
        v60 = &v59[2 * v58];
        do
        {
          if ( *v59 != -16 && *v59 != -8 )
          {
            v61 = v59[1];
            if ( v61 )
            {
              v62 = *(_QWORD *)(v61 + 24);
              if ( v62 )
                j_j___libc_free_0(v62, *(_QWORD *)(v61 + 40) - v62);
              j_j___libc_free_0(v61, 56);
            }
          }
          v59 += 2;
        }
        while ( v60 != v59 );
      }
      j___libc_free_0(v57[4]);
      if ( (unsigned __int64 *)*v57 != v57 + 2 )
        _libc_free(*v57);
      j_j___libc_free_0(v57, 80);
    }
    if ( v102 != v101 )
      _libc_free(v102);
    if ( v99 != v100 )
      _libc_free((unsigned __int64)v99);
    _libc_free(v98);
    _libc_free(v97);
    _libc_free(v96);
    v95[0] = (unsigned __int64)&unk_49EE078;
    sub_16366C0(v95);
    return 0;
  }
  sub_1E290B0((__int64)v89);
  v26 = (unsigned __int64 *)v103;
  v95[0] = (unsigned __int64)&unk_49FB698;
  if ( v103 )
  {
    v27 = *(_DWORD *)(v103 + 48);
    if ( v27 )
    {
      v28 = *(_QWORD **)(v103 + 32);
      v29 = &v28[2 * v27];
      do
      {
        if ( *v28 != -16 && *v28 != -8 )
        {
          v30 = v28[1];
          if ( v30 )
          {
            v31 = *(_QWORD *)(v30 + 24);
            if ( v31 )
            {
              v84 = v28[1];
              j_j___libc_free_0(v31, *(_QWORD *)(v30 + 40) - v31);
              v30 = v84;
            }
            j_j___libc_free_0(v30, 56);
          }
        }
        v28 += 2;
      }
      while ( v29 != v28 );
    }
    j___libc_free_0(v26[4]);
    if ( (unsigned __int64 *)*v26 != v26 + 2 )
      _libc_free(*v26);
    j_j___libc_free_0(v26, 80);
  }
  if ( v102 != v101 )
    _libc_free(v102);
  if ( v99 != v100 )
    _libc_free((unsigned __int64)v99);
  _libc_free(v98);
  _libc_free(v97);
  _libc_free(v96);
  v95[0] = (unsigned __int64)&unk_49EE078;
  sub_16366C0(v95);
LABEL_11:
  v8 = (_QWORD *)a2[41];
  if ( v8 != i )
  {
    while ( v8 + 3 == (_QWORD *)(v8[3] & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v8 = (_QWORD *)v8[1];
      if ( v8 == i )
        return 0;
    }
    v9 = (__int64 *)a2[2];
    v10 = 0;
    v11 = *v9;
    v12 = *(__int64 (**)(void))(*v9 + 40);
    if ( v12 != sub_1D00B00 )
    {
      v10 = (_QWORD *)v12();
      v11 = *(_QWORD *)a2[2];
    }
    v13 = *(__int64 (**)())(v11 + 32);
    v14 = v8[4];
    if ( v13 == sub_1F6A4D0 || (v82 = v13()) == 0 )
    {
      sub_1E1A6B0(v14, "An attempt to perform XRay instrumentation for an unsupported target.", 69);
      return 0;
    }
    v33 = v8[7];
    v34 = (__int64 *)(v14 + 64);
    v35 = v10[1] + 1728LL;
    if ( (*(_BYTE *)(v14 + 46) & 4) != 0 )
    {
      v36 = sub_1E0B640(v33, v35, v34, 0);
      sub_1DD6E10((__int64)v8, (__int64 *)v14, (__int64)v36);
    }
    else
    {
      v54 = (__int64)sub_1E0B640(v33, v35, v34, 0);
      sub_1DD5BA0(v8 + 2, v54);
      v55 = *(_QWORD *)v14;
      v56 = *(_QWORD *)v54;
      *(_QWORD *)(v54 + 8) = v14;
      v55 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v54 = v55 | v56 & 7;
      *(_QWORD *)(v55 + 8) = v54;
      *(_QWORD *)v14 = *(_QWORD *)v14 & 7LL | v54;
    }
    switch ( *(_DWORD *)(a2[1] + 504LL) )
    {
      case 1:
      case 3:
      case 0xA:
      case 0xB:
      case 0xC:
      case 0xD:
      case 0x1D:
        v37 = (_QWORD *)a2[41];
        if ( v37 == i )
          return v82;
        break;
      case 0x12:
        sub_1F6A5A0((__int64)a2, (__int64)v10, 0x100u);
        return v82;
      default:
        sub_1F6A5A0((__int64)a2, (__int64)v10, 1u);
        return v82;
    }
    while ( 1 )
    {
      v38 = v37 + 3;
      v39 = sub_1DD5EE0((__int64)v37);
      if ( v37 + 3 != (_QWORD *)v39 )
        break;
LABEL_76:
      v37 = (_QWORD *)v37[1];
      if ( v37 == i )
        return v82;
    }
    while ( 1 )
    {
      v40 = *(_WORD *)(v39 + 46);
      if ( (v40 & 4) != 0 || (v40 & 8) == 0 )
        v41 = (*(_QWORD *)(*(_QWORD *)(v39 + 16) + 8LL) >> 3) & 1LL;
      else
        LOBYTE(v41) = sub_1E15D00(v39, 8u, 1);
      v42 = (_BYTE)v41 == 0;
      v43 = *(char (__fastcall **)(__int64, __int64))(*v10 + 1008LL);
      v44 = v42 ? 0 : 0x1D;
      if ( v43 == sub_1F3AA70 )
      {
        v45 = *(_WORD *)(v39 + 46);
        if ( (v45 & 4) != 0 || (v45 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v39 + 16) + 8LL) & 8LL) == 0 )
            goto LABEL_71;
        }
        else if ( !sub_1E15D00(v39, 8u, 1) )
        {
          goto LABEL_71;
        }
        v52 = *(_WORD *)(v39 + 46);
        if ( (v52 & 4) != 0 )
        {
          if ( v44 )
          {
            v46 = v37[7];
            v47 = (__int64 *)(v39 + 64);
            v48 = v10[1] + ((unsigned __int64)v44 << 6);
LABEL_89:
            v53 = sub_1E0B640(v46, v48, v47, 0);
            sub_1DD6E10((__int64)v37, (__int64 *)v39, (__int64)v53);
            goto LABEL_74;
          }
          goto LABEL_74;
        }
        if ( (v52 & 8) == 0 )
        {
          if ( !v44 )
            goto LABEL_74;
          v46 = v37[7];
          v47 = (__int64 *)(v39 + 64);
          v48 = v10[1] + ((unsigned __int64)v44 << 6);
LABEL_73:
          v49 = (__int64)sub_1E0B640(v46, v48, v47, 0);
          sub_1DD5BA0(v37 + 2, v49);
          v50 = *(_QWORD *)v39;
          v51 = *(_QWORD *)v49;
          *(_QWORD *)(v49 + 8) = v39;
          v50 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v49 = v50 | v51 & 7;
          *(_QWORD *)(v50 + 8) = v49;
          *(_QWORD *)v39 = *(_QWORD *)v39 & 7LL | v49;
          goto LABEL_74;
        }
        sub_1E15D00(v39, 0x10u, 1);
      }
      else
      {
        v43((__int64)v10, v39);
      }
LABEL_71:
      if ( v44 )
      {
        v46 = v37[7];
        v47 = (__int64 *)(v39 + 64);
        v48 = v10[1] + ((unsigned __int64)v44 << 6);
        if ( (*(_BYTE *)(v39 + 46) & 4) != 0 )
          goto LABEL_89;
        goto LABEL_73;
      }
LABEL_74:
      if ( (*(_BYTE *)v39 & 4) != 0 )
      {
        v39 = *(_QWORD *)(v39 + 8);
        if ( (_QWORD *)v39 == v38 )
          goto LABEL_76;
      }
      else
      {
        while ( (*(_BYTE *)(v39 + 46) & 8) != 0 )
          v39 = *(_QWORD *)(v39 + 8);
        v39 = *(_QWORD *)(v39 + 8);
        if ( (_QWORD *)v39 == v38 )
          goto LABEL_76;
      }
    }
  }
  return 0;
}
