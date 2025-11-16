// Function: sub_26A8540
// Address: 0x26a8540
//
__int64 __fastcall sub_26A8540(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // r15
  __int64 *v13; // r12
  __int64 *v14; // rsi
  __int64 *v15; // r12
  __int64 *v16; // r14
  __int64 v17; // r9
  __int64 v18; // r8
  _QWORD *v19; // r10
  int v20; // r11d
  unsigned int v21; // eax
  _QWORD *v22; // rdi
  __int64 v23; // rcx
  unsigned int v24; // esi
  int v25; // eax
  int v26; // ecx
  __int64 v27; // rax
  __int64 v28; // rdi
  int v29; // edx
  int v30; // r11d
  __int64 *v31; // r12
  __int64 *v32; // r14
  __int64 v33; // r9
  __int64 v34; // r8
  _QWORD *v35; // r10
  int v36; // r11d
  unsigned int v37; // eax
  _QWORD *v38; // rdi
  __int64 v39; // rcx
  unsigned int v40; // esi
  int v41; // eax
  int v42; // ecx
  __int64 v43; // rax
  __int64 v44; // rdi
  int v45; // edx
  int v46; // r11d
  int v48; // eax
  __int64 v49; // r15
  __int64 v50; // rax
  int v51; // eax
  __int64 v52; // r15
  __int64 v53; // rax
  int v54; // eax
  int v55; // ecx
  int v56; // r11d
  __int64 v57; // rax
  __int64 v58; // rdi
  int v59; // eax
  int v60; // ecx
  int v61; // r11d
  __int64 v62; // rax
  __int64 v63; // rdi
  __int64 v64; // [rsp+10h] [rbp-1D0h]
  __int64 v65; // [rsp+10h] [rbp-1D0h]
  _QWORD v67[7]; // [rsp+30h] [rbp-1B0h] BYREF
  unsigned int v68; // [rsp+68h] [rbp-178h]
  _QWORD *v69; // [rsp+70h] [rbp-170h]
  _QWORD v70[5]; // [rsp+80h] [rbp-160h] BYREF
  unsigned int v71; // [rsp+A8h] [rbp-138h]
  _QWORD *v72; // [rsp+B0h] [rbp-130h]
  _QWORD v73[5]; // [rsp+C0h] [rbp-120h] BYREF
  unsigned int v74; // [rsp+E8h] [rbp-F8h]
  char *v75; // [rsp+F0h] [rbp-F0h]
  char v76; // [rsp+100h] [rbp-E0h] BYREF
  __int64 (__fastcall **v77)(); // [rsp+120h] [rbp-C0h]
  __int64 v78; // [rsp+138h] [rbp-A8h]
  unsigned int v79; // [rsp+148h] [rbp-98h]
  _QWORD *v80; // [rsp+150h] [rbp-90h]
  _QWORD v81[5]; // [rsp+160h] [rbp-80h] BYREF
  unsigned int v82; // [rsp+188h] [rbp-58h]
  char *v83; // [rsp+190h] [rbp-50h]
  char v84; // [rsp+1A8h] [rbp-38h] BYREF

  v2 = *(_QWORD *)a1;
  v67[0] = a2 & 0xFFFFFFFFFFFFFFFCLL;
  v67[1] = 0;
  nullsub_1518();
  v3 = sub_26A73D0(v2, a2 & 0xFFFFFFFFFFFFFFFCLL, 0, *(_QWORD *)(a1 + 8), 1, 1);
  if ( !v3 )
    return 0;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = v3;
  v6 = *(_QWORD *)(v3 + 296);
  if ( v6 )
  {
    v7 = *(_QWORD *)(v4 + 296);
    if ( v7 && v6 != v7 )
      goto LABEL_119;
    *(_QWORD *)(v4 + 296) = v6;
  }
  v8 = *(_QWORD *)(v5 + 312);
  if ( !v8 )
    goto LABEL_10;
  v9 = *(_QWORD *)(v4 + 312);
  if ( v9 && v8 != v9 )
LABEL_119:
    BUG();
  *(_QWORD *)(v4 + 312) = v8;
LABEL_10:
  v10 = *(_QWORD *)(v5 + 304);
  if ( v10 )
  {
    v11 = *(_QWORD *)(v4 + 304);
    if ( v10 != v11 && v11 )
      goto LABEL_119;
    *(_QWORD *)(v4 + 304) = v10;
  }
  if ( !*(_BYTE *)(v5 + 241) )
    *(_BYTE *)(v4 + 241) = *(_BYTE *)(v4 + 240);
  v12 = *(__int64 **)(v5 + 280);
  v13 = &v12[*(unsigned int *)(v5 + 288)];
  while ( v13 != v12 )
  {
    v14 = v12++;
    sub_269CCD0(v4 + 248, v14);
  }
  if ( !*(_BYTE *)(v5 + 113) )
    *(_BYTE *)(v4 + 113) = *(_BYTE *)(v4 + 112);
  v15 = *(__int64 **)(v5 + 152);
  v16 = &v15[*(unsigned int *)(v5 + 160)];
  v64 = v4 + 120;
  if ( v15 != v16 )
  {
    while ( 1 )
    {
      v24 = *(_DWORD *)(v4 + 144);
      if ( !v24 )
        break;
      v17 = v24 - 1;
      v18 = *(_QWORD *)(v4 + 128);
      v19 = 0;
      v20 = 1;
      v21 = v17 & (((unsigned int)*v15 >> 9) ^ ((unsigned int)*v15 >> 4));
      v22 = (_QWORD *)(v18 + 8LL * v21);
      v23 = *v22;
      if ( *v15 != *v22 )
      {
        while ( v23 != -4096 )
        {
          if ( v23 != -8192 || v19 )
            v22 = v19;
          v21 = v17 & (v20 + v21);
          v23 = *(_QWORD *)(v18 + 8LL * v21);
          if ( *v15 == v23 )
            goto LABEL_22;
          ++v20;
          v19 = v22;
          v22 = (_QWORD *)(v18 + 8LL * v21);
        }
        v51 = *(_DWORD *)(v4 + 136);
        if ( !v19 )
          v19 = v22;
        ++*(_QWORD *)(v4 + 120);
        v29 = v51 + 1;
        if ( 4 * (v51 + 1) < 3 * v24 )
        {
          if ( v24 - *(_DWORD *)(v4 + 140) - v29 <= v24 >> 3 )
          {
            sub_24FB720(v64, v24);
            v54 = *(_DWORD *)(v4 + 144);
            if ( !v54 )
            {
LABEL_117:
              ++*(_DWORD *)(v4 + 136);
              BUG();
            }
            v55 = v54 - 1;
            v56 = 1;
            v17 = 0;
            v18 = *(_QWORD *)(v4 + 128);
            LODWORD(v57) = (v54 - 1) & (((unsigned int)*v15 >> 9) ^ ((unsigned int)*v15 >> 4));
            v19 = (_QWORD *)(v18 + 8LL * (unsigned int)v57);
            v58 = *v19;
            v29 = *(_DWORD *)(v4 + 136) + 1;
            if ( *v15 != *v19 )
            {
              while ( v58 != -4096 )
              {
                if ( v58 == -8192 && !v17 )
                  v17 = (__int64)v19;
                v57 = v55 & (unsigned int)(v57 + v56);
                v19 = (_QWORD *)(v18 + 8 * v57);
                v58 = *v19;
                if ( *v15 == *v19 )
                  goto LABEL_83;
                ++v56;
              }
LABEL_29:
              if ( v17 )
                v19 = (_QWORD *)v17;
            }
          }
LABEL_83:
          *(_DWORD *)(v4 + 136) = v29;
          if ( *v19 != -4096 )
            --*(_DWORD *)(v4 + 140);
          v52 = *v15;
          *v19 = *v15;
          v53 = *(unsigned int *)(v4 + 160);
          if ( v53 + 1 > (unsigned __int64)*(unsigned int *)(v4 + 164) )
          {
            sub_C8D5F0(v4 + 152, (const void *)(v4 + 168), v53 + 1, 8u, v18, v17);
            v53 = *(unsigned int *)(v4 + 160);
          }
          *(_QWORD *)(*(_QWORD *)(v4 + 152) + 8 * v53) = v52;
          ++*(_DWORD *)(v4 + 160);
          goto LABEL_22;
        }
LABEL_25:
        sub_24FB720(v64, 2 * v24);
        v25 = *(_DWORD *)(v4 + 144);
        if ( !v25 )
          goto LABEL_117;
        v26 = v25 - 1;
        v18 = *(_QWORD *)(v4 + 128);
        LODWORD(v27) = (v25 - 1) & (((unsigned int)*v15 >> 9) ^ ((unsigned int)*v15 >> 4));
        v19 = (_QWORD *)(v18 + 8LL * (unsigned int)v27);
        v28 = *v19;
        v29 = *(_DWORD *)(v4 + 136) + 1;
        if ( *v15 != *v19 )
        {
          v30 = 1;
          v17 = 0;
          while ( v28 != -4096 )
          {
            if ( v28 == -8192 && !v17 )
              v17 = (__int64)v19;
            v27 = v26 & (unsigned int)(v27 + v30);
            v19 = (_QWORD *)(v18 + 8 * v27);
            v28 = *v19;
            if ( *v15 == *v19 )
              goto LABEL_83;
            ++v30;
          }
          goto LABEL_29;
        }
        goto LABEL_83;
      }
LABEL_22:
      if ( v16 == ++v15 )
        goto LABEL_32;
    }
    ++*(_QWORD *)(v4 + 120);
    goto LABEL_25;
  }
LABEL_32:
  if ( !*(_BYTE *)(v5 + 177) )
    *(_BYTE *)(v4 + 177) = *(_BYTE *)(v4 + 176);
  v31 = *(__int64 **)(v5 + 216);
  v32 = &v31[*(unsigned int *)(v5 + 224)];
  v65 = v4 + 184;
  if ( v31 != v32 )
  {
    while ( 1 )
    {
      v40 = *(_DWORD *)(v4 + 208);
      if ( !v40 )
        break;
      v33 = v40 - 1;
      v34 = *(_QWORD *)(v4 + 192);
      v35 = 0;
      v36 = 1;
      v37 = v33 & (((unsigned int)*v31 >> 9) ^ ((unsigned int)*v31 >> 4));
      v38 = (_QWORD *)(v34 + 8LL * v37);
      v39 = *v38;
      if ( *v31 != *v38 )
      {
        while ( v39 != -4096 )
        {
          if ( v39 != -8192 || v35 )
            v38 = v35;
          v37 = v33 & (v36 + v37);
          v39 = *(_QWORD *)(v34 + 8LL * v37);
          if ( *v31 == v39 )
            goto LABEL_37;
          ++v36;
          v35 = v38;
          v38 = (_QWORD *)(v34 + 8LL * v37);
        }
        v48 = *(_DWORD *)(v4 + 200);
        if ( !v35 )
          v35 = v38;
        ++*(_QWORD *)(v4 + 184);
        v45 = v48 + 1;
        if ( 4 * (v48 + 1) < 3 * v40 )
        {
          if ( v40 - *(_DWORD *)(v4 + 204) - v45 <= v40 >> 3 )
          {
            sub_24FB720(v65, v40);
            v59 = *(_DWORD *)(v4 + 208);
            if ( !v59 )
            {
LABEL_118:
              ++*(_DWORD *)(v4 + 200);
              BUG();
            }
            v60 = v59 - 1;
            v61 = 1;
            v33 = 0;
            v34 = *(_QWORD *)(v4 + 192);
            LODWORD(v62) = (v59 - 1) & (((unsigned int)*v31 >> 9) ^ ((unsigned int)*v31 >> 4));
            v35 = (_QWORD *)(v34 + 8LL * (unsigned int)v62);
            v63 = *v35;
            v45 = *(_DWORD *)(v4 + 200) + 1;
            if ( *v31 != *v35 )
            {
              while ( v63 != -4096 )
              {
                if ( !v33 && v63 == -8192 )
                  v33 = (__int64)v35;
                v62 = v60 & (unsigned int)(v62 + v61);
                v35 = (_QWORD *)(v34 + 8 * v62);
                v63 = *v35;
                if ( *v31 == *v35 )
                  goto LABEL_69;
                ++v61;
              }
LABEL_44:
              if ( v33 )
                v35 = (_QWORD *)v33;
            }
          }
LABEL_69:
          *(_DWORD *)(v4 + 200) = v45;
          if ( *v35 != -4096 )
            --*(_DWORD *)(v4 + 204);
          v49 = *v31;
          *v35 = *v31;
          v50 = *(unsigned int *)(v4 + 224);
          if ( v50 + 1 > (unsigned __int64)*(unsigned int *)(v4 + 228) )
          {
            sub_C8D5F0(v4 + 216, (const void *)(v4 + 232), v50 + 1, 8u, v34, v33);
            v50 = *(unsigned int *)(v4 + 224);
          }
          *(_QWORD *)(*(_QWORD *)(v4 + 216) + 8 * v50) = v49;
          ++*(_DWORD *)(v4 + 224);
          goto LABEL_37;
        }
LABEL_40:
        sub_24FB720(v65, 2 * v40);
        v41 = *(_DWORD *)(v4 + 208);
        if ( !v41 )
          goto LABEL_118;
        v42 = v41 - 1;
        v34 = *(_QWORD *)(v4 + 192);
        LODWORD(v43) = (v41 - 1) & (((unsigned int)*v31 >> 9) ^ ((unsigned int)*v31 >> 4));
        v35 = (_QWORD *)(v34 + 8LL * (unsigned int)v43);
        v44 = *v35;
        v45 = *(_DWORD *)(v4 + 200) + 1;
        if ( *v35 != *v31 )
        {
          v46 = 1;
          v33 = 0;
          while ( v44 != -4096 )
          {
            if ( !v33 && v44 == -8192 )
              v33 = (__int64)v35;
            v43 = v42 & (unsigned int)(v43 + v46);
            v35 = (_QWORD *)(v34 + 8 * v43);
            v44 = *v35;
            if ( *v31 == *v35 )
              goto LABEL_69;
            ++v46;
          }
          goto LABEL_44;
        }
        goto LABEL_69;
      }
LABEL_37:
      if ( v32 == ++v31 )
        goto LABEL_47;
    }
    ++*(_QWORD *)(v4 + 184);
    goto LABEL_40;
  }
LABEL_47:
  *(_BYTE *)(v4 + 464) |= *(_BYTE *)(v5 + 464);
  sub_266FF60((__int64)v67, v4 + 88);
  v67[0] = off_49D3CA8;
  v81[0] = off_4A1FCF8;
  if ( v83 != &v84 )
    _libc_free((unsigned __int64)v83);
  sub_C7D6A0(v81[3], v82, 1);
  v77 = off_4A1FC98;
  if ( v80 != v81 )
    _libc_free((unsigned __int64)v80);
  sub_C7D6A0(v78, 8LL * v79, 8);
  v73[0] = off_4A1FC38;
  if ( v75 != &v76 )
    _libc_free((unsigned __int64)v75);
  sub_C7D6A0(v73[3], 8LL * v74, 8);
  v70[0] = off_4A1FBD8;
  if ( v72 != v73 )
    _libc_free((unsigned __int64)v72);
  sub_C7D6A0(v70[3], 8LL * v71, 8);
  v67[2] = off_4A1FB78;
  if ( v69 != v70 )
    _libc_free((unsigned __int64)v69);
  sub_C7D6A0(v67[5], 8LL * v68, 8);
  **(_BYTE **)(a1 + 16) &= *(_BYTE *)(v5 + 241) == *(_BYTE *)(v5 + 240);
  **(_BYTE **)(a1 + 24) &= *(_BYTE *)(v5 + 112) == *(_BYTE *)(v5 + 113);
  **(_BYTE **)(a1 + 24) &= *(_BYTE *)(v5 + 176) == *(_BYTE *)(v5 + 177);
  return 1;
}
