// Function: sub_19AF1A0
// Address: 0x19af1a0
//
__int64 __fastcall sub_19AF1A0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __int64 a9)
{
  int v9; // r15d
  int v10; // r15d
  unsigned int i; // ebx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r8
  unsigned __int64 v17; // rax
  unsigned int v18; // eax
  unsigned __int64 v19; // rax
  char v20; // al
  __int64 v21; // r13
  __int64 v22; // rax
  int v23; // ecx
  __int64 v24; // rsi
  int v25; // ecx
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // r15
  unsigned int j; // ebx
  unsigned __int64 v33; // rdi
  __int64 v34; // rsi
  _QWORD *v35; // r13
  int v36; // ecx
  unsigned int v37; // ecx
  __int64 v38; // rax
  __int64 v39; // rdx
  int v40; // r11d
  __int64 *v41; // r13
  unsigned int v42; // edi
  __int64 *v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 *v47; // rax
  __int64 v48; // rsi
  unsigned __int64 v49; // rcx
  __int64 v50; // rcx
  __int64 v51; // rdi
  int v53; // eax
  __int64 v54; // rdi
  unsigned __int64 v55; // rax
  _QWORD *v56; // rax
  __int64 *v57; // rcx
  __int64 v58; // r9
  int v59; // eax
  __int64 v60; // rdx
  __int64 *v61; // rax
  __int64 v62; // rcx
  unsigned __int64 v63; // rdx
  __int64 v64; // rdx
  unsigned int v65; // edx
  __int64 v66; // rsi
  int v67; // r11d
  __int64 *v68; // r10
  int v69; // r11d
  unsigned int v70; // edx
  __int64 v71; // rsi
  int v72; // eax
  int v73; // edi
  __int64 v74; // rax
  __int64 v75; // [rsp+10h] [rbp-D0h]
  _QWORD *v76; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v77; // [rsp+20h] [rbp-C0h]
  unsigned int v78; // [rsp+28h] [rbp-B8h]
  __int64 v79; // [rsp+28h] [rbp-B8h]
  unsigned int v80; // [rsp+28h] [rbp-B8h]
  __int64 v81; // [rsp+30h] [rbp-B0h]
  int v82; // [rsp+30h] [rbp-B0h]
  __int64 *v83; // [rsp+30h] [rbp-B0h]
  __int64 v84; // [rsp+30h] [rbp-B0h]
  __int64 v85; // [rsp+30h] [rbp-B0h]
  _QWORD *v90; // [rsp+68h] [rbp-78h] BYREF
  __int64 v91; // [rsp+70h] [rbp-70h] BYREF
  __int64 v92; // [rsp+78h] [rbp-68h]
  __int64 v93; // [rsp+80h] [rbp-60h]
  unsigned int v94; // [rsp+88h] [rbp-58h]
  char *v95; // [rsp+90h] [rbp-50h] BYREF
  __int64 v96; // [rsp+98h] [rbp-48h]
  _DWORD v97[16]; // [rsp+A0h] [rbp-40h] BYREF

  v9 = *(_DWORD *)(a2 + 20);
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v10 = v9 & 0xFFFFFFF;
  if ( !v10 )
  {
    v51 = 0;
    return j___libc_free_0(v51);
  }
  for ( i = 0; i != v10; ++i )
  {
    while ( (*(_BYTE *)(a2 + 23) & 0x40) == 0 )
    {
      v14 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v15 = i;
      if ( *(_QWORD *)(a4 + 8) == *(_QWORD *)(v14 + 24LL * i) )
        goto LABEL_7;
LABEL_4:
      if ( v10 == ++i )
        goto LABEL_40;
    }
    v14 = *(_QWORD *)(a2 - 8);
    v15 = i;
    if ( *(_QWORD *)(a4 + 8) != *(_QWORD *)(v14 + 24LL * i) )
      goto LABEL_4;
LABEL_7:
    v16 = *(_QWORD *)(v14 + 8 * v15 + 24LL * *(unsigned int *)(a2 + 56) + 8);
    v90 = (_QWORD *)v16;
    if ( v10 != 1 )
    {
      v17 = sub_157EBA0(v16);
      v18 = sub_15F4D60(v17);
      v16 = (__int64)v90;
      if ( v18 > 1 )
      {
        v81 = (__int64)v90;
        v19 = sub_157EBA0((__int64)v90);
        v16 = v81;
        v20 = *(_BYTE *)(v19 + 16);
        if ( v20 != 34 && v20 != 28 )
        {
          v21 = *(_QWORD *)(a2 + 40);
          v22 = a1[3];
          v23 = *(_DWORD *)(v22 + 24);
          if ( v23 )
          {
            v24 = *(_QWORD *)(v22 + 8);
            v25 = v23 - 1;
            v26 = v25 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
            v27 = (__int64 *)(v24 + 16LL * v26);
            v28 = *v27;
            if ( v21 == *v27 )
            {
LABEL_13:
              v29 = v27[1];
              if ( v29 && v21 == **(_QWORD **)(v29 + 32) )
                goto LABEL_29;
            }
            else
            {
              v72 = 1;
              while ( v28 != -8 )
              {
                v73 = v72 + 1;
                v74 = v25 & (v26 + v72);
                v26 = v74;
                v27 = (__int64 *)(v24 + 16 * v74);
                v28 = *v27;
                if ( v21 == *v27 )
                  goto LABEL_13;
                v72 = v73;
              }
            }
          }
          if ( sub_157F790(*(_QWORD *)(a2 + 40)) )
          {
            v95 = (char *)v97;
            v96 = 0x200000000LL;
            sub_1AAA850(
              v21,
              (unsigned int)&v90,
              1,
              (unsigned int)byte_3F871B3,
              (unsigned int)byte_3F871B3,
              (unsigned int)&v95,
              a1[2],
              a1[3],
              0);
            v35 = *(_QWORD **)v95;
            if ( v95 != (char *)v97 )
              _libc_free((unsigned __int64)v95);
          }
          else
          {
            v97[0] = 16777473;
            v30 = a1[3];
            v95 = (char *)a1[2];
            v96 = v30;
            v82 = v10;
            v78 = i;
            v31 = sub_157EBA0((__int64)v90);
            for ( j = 0; v21 != sub_15F4DF0(v31, j); ++j )
              ;
            v33 = v31;
            v34 = j;
            v10 = v82;
            i = v78;
            v35 = (_QWORD *)sub_1AAC5F0(v33, v34, &v95);
          }
          v16 = (__int64)v90;
          if ( v35 )
          {
            if ( sub_1377F70(a1[5] + 56, (__int64)v90) && !sub_1377F70(a1[5] + 56, *(_QWORD *)(a2 + 40)) )
              sub_1580B80(v35, *(_QWORD *)(a2 + 40));
            v36 = *(_DWORD *)(a2 + 20);
            v90 = v35;
            v37 = v36 & 0xFFFFFFF;
            v10 = v37;
            if ( v37 )
            {
              i = 0;
              v38 = 24LL * *(unsigned int *)(a2 + 56) + 8;
              while ( 1 )
              {
                v39 = a2 - 24LL * v37;
                if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                  v39 = *(_QWORD *)(a2 - 8);
                if ( v35 == *(_QWORD **)(v39 + v38) )
                  break;
                ++i;
                v38 += 8;
                if ( v37 == i )
                  goto LABEL_87;
              }
            }
            else
            {
LABEL_87:
              i = -1;
            }
            v16 = (__int64)v35;
          }
        }
      }
    }
LABEL_29:
    if ( !v94 )
    {
      ++v91;
      goto LABEL_72;
    }
    v40 = 1;
    v41 = 0;
    v42 = (v94 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v43 = (__int64 *)(v92 + 16LL * v42);
    v44 = *v43;
    if ( v16 != *v43 )
    {
      while ( v44 != -8 )
      {
        if ( v44 == -16 && !v41 )
          v41 = v43;
        v42 = (v94 - 1) & (v40 + v42);
        v43 = (__int64 *)(v92 + 16LL * v42);
        v44 = *v43;
        if ( *v43 == v16 )
          goto LABEL_31;
        ++v40;
      }
      if ( !v41 )
        v41 = v43;
      ++v91;
      v53 = v93 + 1;
      if ( 4 * ((int)v93 + 1) < 3 * v94 )
      {
        if ( v94 - HIDWORD(v93) - v53 > v94 >> 3 )
          goto LABEL_53;
        v80 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
        v85 = v16;
        sub_141A900((__int64)&v91, v94);
        if ( !v94 )
        {
LABEL_99:
          LODWORD(v93) = v93 + 1;
          BUG();
        }
        v68 = 0;
        v16 = v85;
        v69 = 1;
        v70 = (v94 - 1) & v80;
        v53 = v93 + 1;
        v41 = (__int64 *)(v92 + 16LL * v70);
        v71 = *v41;
        if ( *v41 == v85 )
          goto LABEL_53;
        while ( v71 != -8 )
        {
          if ( v71 == -16 && !v68 )
            v68 = v41;
          v70 = (v94 - 1) & (v69 + v70);
          v41 = (__int64 *)(v92 + 16LL * v70);
          v71 = *v41;
          if ( *v41 == v85 )
            goto LABEL_53;
          ++v69;
        }
        goto LABEL_76;
      }
LABEL_72:
      v84 = v16;
      sub_141A900((__int64)&v91, 2 * v94);
      if ( !v94 )
        goto LABEL_99;
      v16 = v84;
      v65 = (v94 - 1) & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
      v53 = v93 + 1;
      v41 = (__int64 *)(v92 + 16LL * v65);
      v66 = *v41;
      if ( v84 == *v41 )
        goto LABEL_53;
      v67 = 1;
      v68 = 0;
      while ( v66 != -8 )
      {
        if ( !v68 && v66 == -16 )
          v68 = v41;
        v65 = (v94 - 1) & (v67 + v65);
        v41 = (__int64 *)(v92 + 16LL * v65);
        v66 = *v41;
        if ( *v41 == v84 )
          goto LABEL_53;
        ++v67;
      }
LABEL_76:
      if ( v68 )
        v41 = v68;
LABEL_53:
      LODWORD(v93) = v53;
      if ( *v41 != -8 )
        --HIDWORD(v93);
      *v41 = v16;
      v54 = (__int64)v90;
      v41[1] = 0;
      v55 = sub_157EBA0(v54);
      v56 = (_QWORD *)sub_19A2ED0((__int64)a1, a3, a4, a5, v55 + 24, a6, a7, a8, a9);
      v57 = *(__int64 **)(a4 + 8);
      v58 = (__int64)v56;
      if ( *v56 != *v57 )
      {
        v75 = *v57;
        v76 = v56;
        v83 = *(__int64 **)(a4 + 8);
        v77 = sub_157EBA0((__int64)v90);
        v95 = "tmp";
        LOWORD(v97[0]) = 259;
        v79 = *v83;
        v59 = sub_15FBEB0(v76, 0, v75, 0);
        v58 = sub_15FDBD0(v59, (__int64)v76, v79, (__int64)&v95, v77);
      }
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v60 = *(_QWORD *)(a2 - 8);
      else
        v60 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v61 = (__int64 *)(v60 + 24LL * i);
      if ( *v61 )
      {
        v62 = v61[1];
        v63 = v61[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v63 = v62;
        if ( v62 )
          *(_QWORD *)(v62 + 16) = *(_QWORD *)(v62 + 16) & 3LL | v63;
      }
      *v61 = v58;
      if ( v58 )
      {
        v64 = *(_QWORD *)(v58 + 8);
        v61[1] = v64;
        if ( v64 )
          *(_QWORD *)(v64 + 16) = (unsigned __int64)(v61 + 1) | *(_QWORD *)(v64 + 16) & 3LL;
        v61[2] = (v58 + 8) | v61[2] & 3;
        *(_QWORD *)(v58 + 8) = v61;
      }
      v41[1] = v58;
      goto LABEL_4;
    }
LABEL_31:
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v45 = *(_QWORD *)(a2 - 8);
    else
      v45 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v46 = v43[1];
    v47 = (__int64 *)(v45 + 24LL * i);
    if ( *v47 )
    {
      v48 = v47[1];
      v49 = v47[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v49 = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 16) = *(_QWORD *)(v48 + 16) & 3LL | v49;
    }
    *v47 = v46;
    if ( !v46 )
      goto LABEL_4;
    v50 = *(_QWORD *)(v46 + 8);
    v47[1] = v50;
    if ( v50 )
      *(_QWORD *)(v50 + 16) = (unsigned __int64)(v47 + 1) | *(_QWORD *)(v50 + 16) & 3LL;
    v47[2] = (v46 + 8) | v47[2] & 3;
    *(_QWORD *)(v46 + 8) = v47;
  }
LABEL_40:
  v51 = v92;
  return j___libc_free_0(v51);
}
