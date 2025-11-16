// Function: sub_321CCA0
// Address: 0x321cca0
//
void __fastcall sub_321CCA0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v8; // r12
  char *v9; // r13
  char *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  char *v19; // rdi
  unsigned __int64 v20; // r12
  char *v21; // r13
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rcx
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r9
  char *v29; // rdi
  char *v30; // r13
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  char *v37; // r12
  bool v38; // cf
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  char *v44; // r13
  __int64 v45; // r14
  __int64 v46; // rax
  __int64 v47; // rbx
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // rcx
  int v51; // r10d
  __int64 v52; // r15
  char **v53; // rcx
  __int64 v54; // rbx
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  char v59; // dl
  int v60; // edi
  __int64 v61; // r12
  __int64 v62; // rdx
  __int64 v63; // r8
  char **v64; // [rsp+0h] [rbp-120h]
  unsigned __int64 v65; // [rsp+8h] [rbp-118h]
  char **v66; // [rsp+10h] [rbp-110h]
  __int64 v67; // [rsp+18h] [rbp-108h]
  unsigned __int64 v68; // [rsp+20h] [rbp-100h]
  unsigned __int64 v69; // [rsp+30h] [rbp-F0h]
  char *v70; // [rsp+40h] [rbp-E0h]
  unsigned __int64 i; // [rsp+48h] [rbp-D8h]
  __int64 v72; // [rsp+48h] [rbp-D8h]
  __int64 v73; // [rsp+50h] [rbp-D0h] BYREF
  char *v74; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v75; // [rsp+60h] [rbp-C0h]
  _BYTE v76[48]; // [rsp+68h] [rbp-B8h] BYREF
  char v77; // [rsp+98h] [rbp-88h]
  __int64 v78; // [rsp+A0h] [rbp-80h] BYREF
  char *v79; // [rsp+A8h] [rbp-78h] BYREF
  __int64 v80; // [rsp+B0h] [rbp-70h]
  _BYTE v81[48]; // [rsp+B8h] [rbp-68h] BYREF
  char v82; // [rsp+E8h] [rbp-38h]

  v6 = a2 - a1;
  v68 = a2;
  v67 = a3;
  if ( (__int64)(a2 - a1) <= 1280 )
    return;
  if ( !a3 )
  {
    v69 = a2;
    v66 = (char **)(a1 + 8);
    goto LABEL_38;
  }
  v65 = a1 + 80;
  v66 = (char **)(a1 + 8);
  v64 = (char **)(a1 + 88);
  while ( 2 )
  {
    --v67;
    v8 = a1 + 80 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (v6 >> 4)) >> 1);
    sub_AF47B0(
      (__int64)&v73,
      *(unsigned __int64 **)(*(_QWORD *)(a1 + 80) + 16LL),
      *(unsigned __int64 **)(*(_QWORD *)(a1 + 80) + 24LL));
    v9 = v74;
    sub_AF47B0(
      (__int64)&v78,
      *(unsigned __int64 **)(*(_QWORD *)v8 + 16LL),
      *(unsigned __int64 **)(*(_QWORD *)v8 + 24LL));
    if ( v9 >= v79 )
    {
      sub_AF47B0(
        (__int64)&v73,
        *(unsigned __int64 **)(*(_QWORD *)(a1 + 80) + 16LL),
        *(unsigned __int64 **)(*(_QWORD *)(a1 + 80) + 24LL));
      v30 = v74;
      sub_AF47B0(
        (__int64)&v78,
        *(unsigned __int64 **)(*(_QWORD *)(v68 - 80) + 16LL),
        *(unsigned __int64 **)(*(_QWORD *)(v68 - 80) + 24LL));
      if ( v30 >= v79 )
      {
        sub_AF47B0(
          (__int64)&v73,
          *(unsigned __int64 **)(*(_QWORD *)v8 + 16LL),
          *(unsigned __int64 **)(*(_QWORD *)v8 + 24LL));
        v44 = v74;
        sub_AF47B0(
          (__int64)&v78,
          *(unsigned __int64 **)(*(_QWORD *)(v68 - 80) + 16LL),
          *(unsigned __int64 **)(*(_QWORD *)(v68 - 80) + 24LL));
        v38 = v44 < v79;
        v78 = *(_QWORD *)a1;
        v79 = v81;
        v80 = 0x200000000LL;
        if ( !v38 )
        {
LABEL_7:
          if ( *(_DWORD *)(a1 + 16) )
            sub_3218940((__int64)&v79, v66, v11, v12, v13, v14);
          v82 = *(_BYTE *)(a1 + 72);
          *(_QWORD *)a1 = *(_QWORD *)v8;
          sub_3218940((__int64)v66, (char **)(v8 + 8), v11, v12, v13, v14);
          *(_BYTE *)(a1 + 72) = *(_BYTE *)(v8 + 72);
          *(_QWORD *)v8 = v78;
          sub_3218940(v8 + 8, &v79, v15, v16, v17, v18);
          v19 = v79;
          *(_BYTE *)(v8 + 72) = v82;
          if ( v19 == v81 )
            goto LABEL_11;
          goto LABEL_10;
        }
        v39 = *(unsigned int *)(a1 + 16);
        if ( !(_DWORD)v39 )
        {
LABEL_32:
          v82 = *(_BYTE *)(a1 + 72);
          *(_QWORD *)a1 = *(_QWORD *)(v68 - 80);
          sub_3218940((__int64)v66, (char **)(v68 - 72), v39, v12, v13, v14);
          *(_BYTE *)(a1 + 72) = *(_BYTE *)(v68 - 8);
          *(_QWORD *)(v68 - 80) = v78;
          sub_3218940(v68 - 72, &v79, v40, v41, v42, v43);
          v19 = v79;
          *(_BYTE *)(v68 - 8) = v82;
          if ( v19 == v81 )
            goto LABEL_11;
          goto LABEL_10;
        }
LABEL_36:
        sub_3218940((__int64)&v79, v66, v39, v12, v13, v14);
        goto LABEL_32;
      }
      v78 = *(_QWORD *)a1;
      v79 = v81;
      v80 = 0x200000000LL;
    }
    else
    {
      sub_AF47B0(
        (__int64)&v73,
        *(unsigned __int64 **)(*(_QWORD *)v8 + 16LL),
        *(unsigned __int64 **)(*(_QWORD *)v8 + 24LL));
      v10 = v74;
      sub_AF47B0(
        (__int64)&v78,
        *(unsigned __int64 **)(*(_QWORD *)(v68 - 80) + 16LL),
        *(unsigned __int64 **)(*(_QWORD *)(v68 - 80) + 24LL));
      if ( v10 < v79 )
      {
        v78 = *(_QWORD *)a1;
        v79 = v81;
        v80 = 0x200000000LL;
        goto LABEL_7;
      }
      sub_AF47B0(
        (__int64)&v73,
        *(unsigned __int64 **)(*(_QWORD *)(a1 + 80) + 16LL),
        *(unsigned __int64 **)(*(_QWORD *)(a1 + 80) + 24LL));
      v37 = v74;
      sub_AF47B0(
        (__int64)&v78,
        *(unsigned __int64 **)(*(_QWORD *)(v68 - 80) + 16LL),
        *(unsigned __int64 **)(*(_QWORD *)(v68 - 80) + 24LL));
      v38 = v37 < v79;
      v78 = *(_QWORD *)a1;
      v79 = v81;
      v80 = 0x200000000LL;
      if ( v38 )
      {
        v39 = *(unsigned int *)(a1 + 16);
        if ( !(_DWORD)v39 )
          goto LABEL_32;
        goto LABEL_36;
      }
    }
    v32 = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)v32 )
      sub_3218940((__int64)&v79, v66, v31, v32, v13, v14);
    v82 = *(_BYTE *)(a1 + 72);
    *(_QWORD *)a1 = *(_QWORD *)(a1 + 80);
    sub_3218940((__int64)v66, v64, v31, v32, v13, v14);
    *(_BYTE *)(a1 + 72) = *(_BYTE *)(a1 + 152);
    *(_QWORD *)(a1 + 80) = v78;
    sub_3218940((__int64)v64, &v79, v33, v34, v35, v36);
    v19 = v79;
    *(_BYTE *)(a1 + 152) = v82;
    if ( v19 == v81 )
      goto LABEL_11;
LABEL_10:
    _libc_free((unsigned __int64)v19);
LABEL_11:
    v20 = v68;
    for ( i = v65; ; i += 80LL )
    {
      v69 = i;
      sub_AF47B0(
        (__int64)&v73,
        *(unsigned __int64 **)(*(_QWORD *)i + 16LL),
        *(unsigned __int64 **)(*(_QWORD *)i + 24LL));
      v70 = v74;
      sub_AF47B0(
        (__int64)&v78,
        *(unsigned __int64 **)(*(_QWORD *)a1 + 16LL),
        *(unsigned __int64 **)(*(_QWORD *)a1 + 24LL));
      if ( v70 < v79 )
        continue;
      do
      {
        v20 -= 80LL;
        sub_AF47B0(
          (__int64)&v73,
          *(unsigned __int64 **)(*(_QWORD *)a1 + 16LL),
          *(unsigned __int64 **)(*(_QWORD *)a1 + 24LL));
        v21 = v74;
        sub_AF47B0(
          (__int64)&v78,
          *(unsigned __int64 **)(*(_QWORD *)v20 + 16LL),
          *(unsigned __int64 **)(*(_QWORD *)v20 + 24LL));
      }
      while ( v21 < v79 );
      if ( i >= v20 )
        break;
      v24 = i;
      v25 = i + 8;
      v78 = *(_QWORD *)i;
      v79 = v81;
      v80 = 0x200000000LL;
      if ( *(_DWORD *)(i + 16) )
      {
        sub_3218940((__int64)&v79, (char **)(i + 8), v22, i, v23, v25);
        v25 = i + 8;
      }
      v82 = *(_BYTE *)(i + 72);
      *(_QWORD *)i = *(_QWORD *)v20;
      sub_3218940(v25, (char **)(v20 + 8), v22, v24, v20 + 8, v25);
      *(_BYTE *)(i + 72) = *(_BYTE *)(v20 + 72);
      *(_QWORD *)v20 = v78;
      sub_3218940(v20 + 8, &v79, v26, v27, v20 + 8, v28);
      v29 = v79;
      *(_BYTE *)(v20 + 72) = v82;
      if ( v29 != v81 )
        _libc_free((unsigned __int64)v29);
    }
    sub_321CCA0(i, v68, v67);
    v6 = i - a1;
    if ( (__int64)(i - a1) > 1280 )
    {
      if ( v67 )
      {
        v68 = i;
        continue;
      }
LABEL_38:
      v72 = 0xCCCCCCCCCCCCCCCDLL * (v6 >> 4);
      v45 = (v72 - 2) >> 1;
      v46 = a1;
      v47 = a1 + 80 * v45 + 8;
      v48 = v46;
      while ( 1 )
      {
        v50 = *(_QWORD *)(v47 - 8);
        v51 = *(_DWORD *)(v47 + 8);
        v74 = v76;
        v75 = 0x200000000LL;
        v73 = v50;
        if ( v51 )
        {
          sub_3218940((__int64)&v74, (char **)v47, a3, v50, (__int64)&v74, a6);
          v49 = *(unsigned __int8 *)(v47 + 64);
          a6 = (unsigned int)v75;
          v79 = v81;
          v77 = v49;
          v78 = v73;
          v80 = 0x200000000LL;
          if ( (_DWORD)v75 )
          {
            sub_3218940((__int64)&v79, &v74, v49, v73, (__int64)&v74, (unsigned int)v75);
            LOBYTE(v49) = v77;
          }
        }
        else
        {
          v78 = v50;
          LOBYTE(v49) = *(_BYTE *)(v47 + 64);
          v80 = 0x200000000LL;
          v77 = v49;
          v79 = v81;
        }
        v82 = v49;
        sub_321AA10(v48, v45, v72, (__int64)&v78, a5, a6);
        if ( v79 != v81 )
          _libc_free((unsigned __int64)v79);
        if ( !v45 )
          break;
        --v45;
        if ( v74 != v76 )
          _libc_free((unsigned __int64)v74);
        v47 -= 80;
      }
      if ( v74 != v76 )
        _libc_free((unsigned __int64)v74);
      v52 = v48;
      v53 = &v79;
      v54 = v69 - 72;
      do
      {
        v62 = *(_QWORD *)(v54 - 8);
        v63 = *(unsigned int *)(v54 + 8);
        v74 = v76;
        v75 = 0x200000000LL;
        v73 = v62;
        if ( (_DWORD)v63 )
          sub_3218940((__int64)&v74, (char **)v54, v62, (__int64)v53, v63, a6);
        v77 = *(_BYTE *)(v54 + 64);
        v55 = *(_QWORD *)v52;
        *(_QWORD *)(v54 - 8) = *(_QWORD *)v52;
        sub_3218940(v54, v66, v55, (__int64)v53, v63, a6);
        v59 = *(_BYTE *)(v52 + 72);
        v80 = 0x200000000LL;
        v60 = v75;
        *(_BYTE *)(v54 + 64) = v59;
        v79 = v81;
        v78 = v73;
        if ( v60 )
          sub_3218940((__int64)&v79, &v74, v73, v56, v57, v58);
        v61 = v54 - 8 - v52;
        v82 = v77;
        sub_321AA10(v52, 0, 0xCCCCCCCCCCCCCCCDLL * (v61 >> 4), (__int64)&v78, v57, v58);
        if ( v79 != v81 )
          _libc_free((unsigned __int64)v79);
        if ( v74 != v76 )
          _libc_free((unsigned __int64)v74);
        v54 -= 80;
      }
      while ( v61 > 80 );
    }
    break;
  }
}
