// Function: sub_37D7660
// Address: 0x37d7660
//
__int64 __fastcall sub_37D7660(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rax
  __int64 v8; // r8
  char v9; // al
  __int64 v10; // rdx
  char v11; // r10
  __int64 v12; // r9
  int v13; // edi
  unsigned int v14; // esi
  __int64 *v15; // rcx
  __int64 v16; // r11
  __int64 v17; // rsi
  __int64 v18; // rbx
  __int64 v19; // rcx
  int v20; // eax
  __int64 v21; // rdi
  int v22; // r8d
  unsigned int v23; // esi
  __int64 *v24; // rcx
  __int64 v25; // r9
  __int64 v26; // rdx
  __int64 v27; // r12
  int v28; // eax
  unsigned int v29; // eax
  int v30; // ecx
  int v31; // edx
  unsigned int v32; // r12d
  int v34; // esi
  unsigned int v35; // edi
  unsigned int v36; // edx
  __int64 v37; // rcx
  int v38; // r11d
  void *v39; // r10
  unsigned int v40; // ecx
  unsigned int v41; // edi
  unsigned int v42; // esi
  __int64 v43; // rdi
  __int64 v44; // rcx
  int v45; // ecx
  __int64 v46; // rdi
  int v47; // ecx
  unsigned int v48; // r12d
  char **v49; // r8
  __int64 *v50; // rsi
  unsigned int v51; // eax
  __int64 v52; // rdx
  int v53; // edi
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  int v58; // esi
  unsigned int v59; // eax
  void *v60; // r14
  int v61; // ebx
  size_t v62; // r13
  __int64 v63; // rax
  unsigned __int64 v64; // rdx
  int v65; // r10d
  int v66; // ebx
  __int64 *v71; // [rsp+28h] [rbp-F8h]
  __int64 *v72; // [rsp+40h] [rbp-E0h]
  char *v74; // [rsp+50h] [rbp-D0h]
  __int64 v75; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v76; // [rsp+68h] [rbp-B8h]
  __int64 *v77; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v78; // [rsp+78h] [rbp-A8h]
  char *v79; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v80; // [rsp+88h] [rbp-98h]
  __int64 v81; // [rsp+90h] [rbp-90h]
  __int64 v82; // [rsp+98h] [rbp-88h]
  __int64 v83; // [rsp+A0h] [rbp-80h]
  __int64 v84; // [rsp+A8h] [rbp-78h]
  void *src; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v86; // [rsp+B8h] [rbp-68h]
  _BYTE v87[96]; // [rsp+C0h] [rbp-60h] BYREF

  v7 = (__int64 *)&v77;
  v75 = 0;
  v76 = 1;
  do
  {
    *(_DWORD *)v7 = -1;
    v7 = (__int64 *)((char *)v7 + 4);
  }
  while ( v7 != (__int64 *)&v79 );
  v8 = *(_QWORD *)a6;
  v9 = *(_BYTE *)(a4 + 8);
  v10 = **(_QWORD **)a6;
  v11 = v9 & 1;
  if ( (v9 & 1) != 0 )
  {
    v12 = a4 + 16;
    v13 = 15;
  }
  else
  {
    v43 = *(unsigned int *)(a4 + 24);
    v12 = *(_QWORD *)(a4 + 16);
    if ( !(_DWORD)v43 )
      goto LABEL_79;
    v13 = v43 - 1;
  }
  v14 = v13 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v15 = (__int64 *)(v12 + 16LL * v14);
  v16 = *v15;
  if ( v10 != *v15 )
  {
    v47 = 1;
    while ( v16 != -4096 )
    {
      v66 = v47 + 1;
      v14 = v13 & (v47 + v14);
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v10 == *v15 )
        goto LABEL_6;
      v47 = v66;
    }
    if ( v11 )
    {
      v46 = 256;
      goto LABEL_80;
    }
    v43 = *(unsigned int *)(a4 + 24);
LABEL_79:
    v46 = 16 * v43;
LABEL_80:
    v15 = (__int64 *)(v12 + v46);
  }
LABEL_6:
  v17 = 256;
  if ( !v11 )
    v17 = 16LL * *(unsigned int *)(a4 + 24);
  if ( v15 == (__int64 *)(v12 + v17) )
  {
LABEL_29:
    v32 = 0;
    goto LABEL_30;
  }
  v18 = v15[1];
  v19 = *(unsigned int *)(a6 + 8);
  v71 = (__int64 *)(v8 + 8 * v19);
  if ( (__int64 *)v8 != v71 )
  {
    v72 = (__int64 *)(v8 + 8);
    v20 = v9 & 1;
    if ( !v20 )
      goto LABEL_44;
LABEL_11:
    v21 = a4 + 16;
    v22 = 15;
LABEL_12:
    v23 = v22 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v24 = (__int64 *)(v21 + 16LL * v23);
    v25 = *v24;
    if ( v10 != *v24 )
    {
      v45 = 1;
      while ( v25 != -4096 )
      {
        v65 = v45 + 1;
        v23 = v22 & (v45 + v23);
        v24 = (__int64 *)(v21 + 16LL * v23);
        v25 = *v24;
        if ( *v24 == v10 )
          goto LABEL_13;
        v45 = v65;
      }
      if ( !(_BYTE)v20 )
      {
        v37 = *(unsigned int *)(a4 + 24);
LABEL_72:
        v44 = 16 * v37;
        goto LABEL_73;
      }
      v44 = 256;
LABEL_73:
      v24 = (__int64 *)(v21 + v44);
    }
LABEL_13:
    v26 = 256;
    if ( !(_BYTE)v20 )
      v26 = 16LL * *(unsigned int *)(a4 + 24);
    if ( v24 == (__int64 *)(v21 + v26) )
      goto LABEL_29;
    v27 = v24[1];
    v28 = *(_DWORD *)(v27 + 56);
    if ( v28 == 3 || v28 == 2 && !*(_DWORD *)(v27 + 32) && *(_DWORD *)(v27 + 36) != *(_DWORD *)(a3 + 24) )
      goto LABEL_29;
    if ( !(unsigned __int8)sub_AF65F0(
                             *(_QWORD *)(v18 + 40),
                             *(_BYTE *)(v18 + 48),
                             *(_QWORD *)(v27 + 40),
                             *(_BYTE *)(v27 + 48)) )
      goto LABEL_29;
    LODWORD(v79) = 0;
    if ( !*(_BYTE *)(v18 + 49) )
    {
LABEL_20:
      v29 = (unsigned int)v79;
      if ( (_DWORD)v79 )
        goto LABEL_42;
      goto LABEL_21;
    }
    while ( 1 )
    {
      v36 = sub_AF4EB0(*(_QWORD *)(v18 + 40));
      v29 = (unsigned int)v79;
      if ( (unsigned int)v79 >= v36 )
      {
LABEL_42:
        if ( v71 == v72 )
          break;
        v10 = *v72++;
        LOBYTE(v20) = *(_BYTE *)(a4 + 8) & 1;
        if ( (_BYTE)v20 )
          goto LABEL_11;
LABEL_44:
        v37 = *(unsigned int *)(a4 + 24);
        v21 = *(_QWORD *)(a4 + 16);
        if ( !(_DWORD)v37 )
          goto LABEL_72;
        v22 = v37 - 1;
        goto LABEL_12;
      }
LABEL_21:
      v30 = *(_DWORD *)(v27 + 32);
      if ( *(_DWORD *)(v27 + 56) == 2 && !v30 )
      {
        sub_37D7460((__int64)&src, (__int64)&v75, (int *)&v79);
        LODWORD(v79) = (_DWORD)v79 + 1;
        goto LABEL_40;
      }
      if ( *(_DWORD *)(v18 + 32) )
        v31 = *(_DWORD *)(v18 + 4LL * v29);
      else
        v31 = dword_5051178[0];
      if ( v30 )
        v19 = *(unsigned int *)(v27 + 4LL * v29);
      else
        v19 = dword_5051178[0];
      if ( (_DWORD)v19 != v31 )
      {
        if ( (v31 & 1) != 0 && v31 != dword_5051178[0] || (v19 & 1) != 0 && (_DWORD)v19 != dword_5051178[0] )
          goto LABEL_29;
        if ( (v76 & 1) != 0 )
        {
          v34 = 3;
          v8 = (__int64)&v77;
        }
        else
        {
          v42 = v78;
          v8 = (__int64)v77;
          if ( !v78 )
          {
            v40 = v76;
            ++v75;
            src = 0;
            v41 = ((unsigned int)v76 >> 1) + 1;
            goto LABEL_63;
          }
          v34 = v78 - 1;
        }
        v19 = v34 & (37 * v29);
        v12 = v8 + 4 * v19;
        v35 = *(_DWORD *)v12;
        if ( v29 != *(_DWORD *)v12 )
        {
          v38 = 1;
          v39 = 0;
          while ( v35 != -1 )
          {
            if ( v35 != -2 || v39 )
              v12 = (__int64)v39;
            v19 = v34 & (unsigned int)(v38 + v19);
            v35 = *(_DWORD *)(v8 + 4LL * (unsigned int)v19);
            if ( v29 == v35 )
              goto LABEL_38;
            ++v38;
            v39 = (void *)v12;
            v12 = v8 + 4LL * (unsigned int)v19;
          }
          v40 = v76;
          if ( !v39 )
            v39 = (void *)v12;
          ++v75;
          src = v39;
          v41 = ((unsigned int)v76 >> 1) + 1;
          if ( (v76 & 1) != 0 )
          {
            v8 = 4 * v41;
            v42 = 4;
            if ( (unsigned int)v8 < 0xC )
              goto LABEL_52;
LABEL_64:
            v42 *= 2;
            goto LABEL_65;
          }
          v42 = v78;
LABEL_63:
          v8 = 4 * v41;
          if ( (unsigned int)v8 >= 3 * v42 )
            goto LABEL_64;
LABEL_52:
          if ( v42 - HIDWORD(v76) - v41 <= v42 >> 3 )
          {
LABEL_65:
            sub_B47550((__int64)&v75, v42);
            sub_37C0030((__int64)&v75, (int *)&v79, &src);
            v29 = (unsigned int)v79;
            v40 = v76;
          }
          v19 = (2 * (v40 >> 1) + 2) | v40 & 1;
          LODWORD(v76) = v19;
          if ( *(_DWORD *)src != -1 )
            --HIDWORD(v76);
          *(_DWORD *)src = v29;
          v29 = (unsigned int)v79;
          goto LABEL_39;
        }
LABEL_38:
        v29 = v35;
      }
LABEL_39:
      LODWORD(v79) = v29 + 1;
LABEL_40:
      if ( !*(_BYTE *)(v18 + 49) )
        goto LABEL_20;
    }
  }
  v48 = 0;
  src = v87;
  v86 = 0xC00000000LL;
  while ( 1 )
  {
    v59 = 1;
    if ( *(_BYTE *)(v18 + 49) )
      v59 = sub_AF4EB0(*(_QWORD *)(v18 + 40));
    if ( v59 <= v48 )
      break;
    if ( (v76 & 1) != 0 )
    {
      v49 = &v79;
      v50 = (__int64 *)&v77;
      v19 = 3;
    }
    else
    {
      v50 = v77;
      v52 = v78;
      v49 = (char **)((char *)v77 + 4 * v78);
      if ( !v78 )
        goto LABEL_102;
      v19 = v78 - 1;
    }
    v51 = v19 & (37 * v48);
    v52 = (__int64)v50 + 4 * v51;
    v53 = *(_DWORD *)v52;
    if ( *(_DWORD *)v52 != v48 )
    {
      v52 = 1;
      while ( v53 != -1 )
      {
        v12 = (unsigned int)(v52 + 1);
        v51 = v19 & (v52 + v51);
        v52 = (__int64)v50 + 4 * v51;
        v53 = *(_DWORD *)v52;
        if ( *(_DWORD *)v52 == v48 )
          goto LABEL_92;
        v52 = (unsigned int)v12;
      }
LABEL_102:
      if ( *(_DWORD *)(v18 + 32) )
        v58 = *(_DWORD *)(v18 + 4LL * v48);
      else
        v58 = dword_5051178[0];
      goto LABEL_95;
    }
LABEL_92:
    if ( (char **)v52 == v49 )
      goto LABEL_102;
    v74 = sub_37C4690(a1, v48, a3, a4, a5, a6);
    if ( !(_BYTE)v54 )
    {
      v32 = 0;
      goto LABEL_114;
    }
    LOBYTE(v84) = 0;
    v79 = v74;
    v58 = sub_37C57D0((__int64)(a1 + 271), v48, v54, v55, v56, v57, (__int64)v74, v80, v81, v82, v83, 0);
LABEL_95:
    ++v48;
    sub_37BC2A0((__int64)&src, v58, v52, v19, (__int64)v49, v12);
  }
  v60 = src;
  v61 = v86;
  v62 = 4LL * (unsigned int)v86;
  v63 = *(unsigned int *)(a2 + 8);
  v64 = (unsigned int)v86 + v63;
  if ( v64 > *(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v64, 4u, v8, v12);
    v63 = *(unsigned int *)(a2 + 8);
  }
  if ( v62 )
  {
    memcpy((void *)(*(_QWORD *)a2 + 4 * v63), v60, v62);
    LODWORD(v63) = *(_DWORD *)(a2 + 8);
  }
  v32 = 1;
  *(_DWORD *)(a2 + 8) = v61 + v63;
LABEL_114:
  if ( src != v87 )
    _libc_free((unsigned __int64)src);
LABEL_30:
  if ( (v76 & 1) == 0 )
    sub_C7D6A0((__int64)v77, 4LL * v78, 4);
  return v32;
}
