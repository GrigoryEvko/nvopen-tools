// Function: sub_1D0E9E0
// Address: 0x1d0e9e0
//
void __fastcall sub_1D0E9E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rcx
  int v9; // edx
  unsigned __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // rax
  _QWORD *v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rbx
  __int64 *v18; // r14
  __int64 *v19; // r13
  int v20; // r8d
  char v21; // dl
  __int64 v22; // rsi
  __int64 *v23; // rax
  __int64 *v24; // rdi
  __int64 *v25; // rcx
  __int64 v26; // r15
  __int64 v27; // rax
  unsigned __int16 v28; // cx
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  int v31; // r8d
  int v32; // r9d
  __int64 *v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // r13
  int v36; // eax
  unsigned int *v37; // rax
  __int64 v38; // r15
  __int64 v39; // rax
  _BYTE *v40; // rdx
  __int64 v41; // rbx
  int v42; // eax
  _BYTE *v43; // r8
  int v44; // esi
  _BYTE *v45; // rdi
  __int64 *v46; // rax
  __int64 v47; // rax
  __int64 v48; // r10
  unsigned __int16 v49; // cx
  unsigned __int64 v50; // rdx
  __int64 v51; // rdx
  int v52; // edx
  unsigned int *v53; // rdx
  __int64 v54; // r14
  __int64 v55; // r13
  __int64 v56; // r12
  __int64 v57; // rbx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rbx
  unsigned __int64 v62; // rdi
  unsigned __int64 v63; // rdi
  __int64 v64; // [rsp+8h] [rbp-3F8h]
  __int64 v65; // [rsp+20h] [rbp-3E0h]
  __int64 v67; // [rsp+30h] [rbp-3D0h] BYREF
  int v68; // [rsp+38h] [rbp-3C8h]
  _BYTE *v69; // [rsp+40h] [rbp-3C0h] BYREF
  __int64 v70; // [rsp+48h] [rbp-3B8h]
  _BYTE v71[64]; // [rsp+50h] [rbp-3B0h] BYREF
  __int64 v72; // [rsp+90h] [rbp-370h] BYREF
  __int64 *v73; // [rsp+98h] [rbp-368h]
  __int64 *v74; // [rsp+A0h] [rbp-360h]
  __int64 v75; // [rsp+A8h] [rbp-358h]
  int v76; // [rsp+B0h] [rbp-350h]
  _QWORD v77[33]; // [rsp+B8h] [rbp-348h] BYREF
  _QWORD *v78; // [rsp+1C0h] [rbp-240h] BYREF
  __int64 v79; // [rsp+1C8h] [rbp-238h]
  _QWORD v80[70]; // [rsp+1D0h] [rbp-230h] BYREF

  v6 = a1[78];
  v7 = *(_QWORD *)(v6 + 200);
  v8 = v6 + 192;
  if ( v7 == v6 + 192 )
    goto LABEL_7;
  v9 = 0;
  do
  {
    if ( !v7 )
    {
      MEMORY[0x1C] = 0;
      BUG();
    }
    *(_DWORD *)(v7 + 20) = -1;
    v7 = *(_QWORD *)(v7 + 8);
    ++v9;
  }
  while ( v7 != v8 );
  v10 = (unsigned int)(2 * v9);
  v11 = 0xF0F0F0F0F0F0F0F1LL;
  v12 = a1[6];
  v13 = a1[8] - v12;
  if ( v10 > 0xF0F0F0F0F0F0F0F1LL * (v13 >> 4) )
  {
    v54 = a1[7];
    v55 = 272 * v10;
    v65 = v54 - v12;
    if ( 2 * v9 )
    {
      v56 = sub_22077B0(272 * v10);
      if ( v12 == v54 )
      {
LABEL_91:
        v61 = a1[7];
        v54 = a1[6];
        if ( v61 == v54 )
        {
          v13 = a1[8] - v54;
        }
        else
        {
          do
          {
            v62 = *(_QWORD *)(v54 + 112);
            if ( v62 != v54 + 128 )
              _libc_free(v62);
            v63 = *(_QWORD *)(v54 + 32);
            if ( v63 != v54 + 48 )
              _libc_free(v63);
            v54 += 272;
          }
          while ( v61 != v54 );
          v54 = a1[6];
          v13 = a1[8] - v54;
        }
        goto LABEL_88;
      }
    }
    else
    {
      v56 = 0;
      if ( v12 == v54 )
      {
LABEL_88:
        if ( v54 )
          j_j___libc_free_0(v54, v13);
        a1[6] = v56;
        v6 = a1[78];
        a1[7] = v56 + v65;
        a1[8] = v55 + v56;
        goto LABEL_7;
      }
    }
    v57 = v56;
    do
    {
      if ( v57 )
      {
        *(_QWORD *)v57 = *(_QWORD *)v12;
        *(_QWORD *)(v57 + 8) = *(_QWORD *)(v12 + 8);
        *(_QWORD *)(v57 + 16) = *(_QWORD *)(v12 + 16);
        v58 = *(_QWORD *)(v12 + 24);
        *(_DWORD *)(v57 + 40) = 0;
        *(_QWORD *)(v57 + 24) = v58;
        *(_QWORD *)(v57 + 32) = v57 + 48;
        *(_DWORD *)(v57 + 44) = 4;
        v59 = *(unsigned int *)(v12 + 40);
        if ( (_DWORD)v59 )
          sub_1D0B890(v57 + 32, v12 + 32, v59, v11, a5, a6);
        *(_DWORD *)(v57 + 120) = 0;
        *(_QWORD *)(v57 + 112) = v57 + 128;
        *(_DWORD *)(v57 + 124) = 4;
        if ( *(_DWORD *)(v12 + 120) )
          sub_1D0B890(v57 + 112, v12 + 112, v59, v11, a5, a6);
        *(_DWORD *)(v57 + 192) = *(_DWORD *)(v12 + 192);
        *(_DWORD *)(v57 + 196) = *(_DWORD *)(v12 + 196);
        *(_DWORD *)(v57 + 200) = *(_DWORD *)(v12 + 200);
        *(_DWORD *)(v57 + 204) = *(_DWORD *)(v12 + 204);
        *(_DWORD *)(v57 + 208) = *(_DWORD *)(v12 + 208);
        *(_DWORD *)(v57 + 212) = *(_DWORD *)(v12 + 212);
        *(_DWORD *)(v57 + 216) = *(_DWORD *)(v12 + 216);
        *(_DWORD *)(v57 + 220) = *(_DWORD *)(v12 + 220);
        *(_WORD *)(v57 + 224) = *(_WORD *)(v12 + 224);
        *(_WORD *)(v57 + 226) = *(_WORD *)(v12 + 226);
        *(_WORD *)(v57 + 228) = *(_WORD *)(v12 + 228);
        *(_DWORD *)(v57 + 232) = *(_DWORD *)(v12 + 232);
        *(_BYTE *)(v57 + 236) = *(_BYTE *)(v12 + 236) & 3 | *(_BYTE *)(v57 + 236) & 0xFC;
        *(_DWORD *)(v57 + 240) = *(_DWORD *)(v12 + 240);
        *(_DWORD *)(v57 + 244) = *(_DWORD *)(v12 + 244);
        *(_DWORD *)(v57 + 248) = *(_DWORD *)(v12 + 248);
        *(_DWORD *)(v57 + 252) = *(_DWORD *)(v12 + 252);
        *(_QWORD *)(v57 + 256) = *(_QWORD *)(v12 + 256);
        *(_QWORD *)(v57 + 264) = *(_QWORD *)(v12 + 264);
      }
      v12 += 272;
      v57 += 272;
    }
    while ( v54 != v12 );
    goto LABEL_91;
  }
  v6 = a1[78];
LABEL_7:
  v76 = 0;
  v73 = v77;
  v74 = v77;
  v14 = *(_QWORD *)(v6 + 176);
  v15 = v80;
  v78 = v80;
  v79 = 0x4000000001LL;
  v80[0] = v14;
  v77[0] = v14;
  v75 = 0x100000020LL;
  v72 = 1;
  v69 = v71;
  v70 = 0x800000000LL;
  v16 = 1;
  while ( 1 )
  {
    v17 = v15[v16 - 1];
    LODWORD(v79) = v16 - 1;
    v18 = *(__int64 **)(v17 + 32);
    v19 = &v18[5 * *(unsigned int *)(v17 + 56)];
    if ( v18 != v19 )
    {
      while ( 1 )
      {
        v22 = *v18;
        v23 = v73;
        if ( v74 == v73 )
        {
          v24 = &v73[HIDWORD(v75)];
          v20 = HIDWORD(v75);
          if ( v73 != v24 )
          {
            v25 = 0;
            while ( v22 != *v23 )
            {
              if ( *v23 == -2 )
                v25 = v23;
              if ( v24 == ++v23 )
              {
                if ( !v25 )
                  goto LABEL_69;
                *v25 = v22;
                --v76;
                ++v72;
                goto LABEL_21;
              }
            }
            goto LABEL_11;
          }
LABEL_69:
          if ( HIDWORD(v75) < (unsigned int)v75 )
            break;
        }
        sub_16CCBA0((__int64)&v72, v22);
        if ( v21 )
        {
LABEL_21:
          v26 = *v18;
          v27 = (unsigned int)v79;
          if ( (unsigned int)v79 >= HIDWORD(v79) )
          {
            sub_16CD150((__int64)&v78, v80, 0, 8, v20, a6);
            v27 = (unsigned int)v79;
          }
          v18 += 5;
          v78[v27] = v26;
          LODWORD(v79) = v79 + 1;
          if ( v19 == v18 )
            goto LABEL_24;
        }
        else
        {
LABEL_11:
          v18 += 5;
          if ( v19 == v18 )
            goto LABEL_24;
        }
      }
      v20 = ++HIDWORD(v75);
      *v24 = v22;
      ++v72;
      goto LABEL_21;
    }
LABEL_24:
    v28 = *(_WORD *)(v17 + 24);
    v29 = (0x7FF0007FF22uLL >> v28) & 1;
    if ( v28 >= 0x2Bu )
      LOBYTE(v29) = 0;
    if ( v28 != 209 && !(_BYTE)v29 && *(_DWORD *)(v17 + 28) == -1 )
      break;
    v16 = v79;
    if ( !(_DWORD)v79 )
      goto LABEL_48;
LABEL_29:
    v15 = v78;
  }
  v30 = sub_1D0E6F0(a1, v17);
  v33 = a1;
  v34 = v17;
  v35 = v30;
  v36 = *(_DWORD *)(v17 + 56);
  if ( v36 )
  {
    while ( 1 )
    {
      v37 = (unsigned int *)(*(_QWORD *)(v34 + 32) + 40LL * (unsigned int)(v36 - 1));
      v34 = *(_QWORD *)v37;
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v37 + 40LL) + 16LL * v37[2]) != 111 )
        break;
      *(_DWORD *)(v34 + 28) = *(_DWORD *)(v35 + 192);
      if ( *(__int16 *)(v34 + 24) < 0
        && (*(_BYTE *)(*(_QWORD *)(a1[2] + 8) + ((__int64)~*(__int16 *)(v34 + 24) << 6) + 8) & 0x10) != 0 )
      {
        *(_BYTE *)(v35 + 228) |= 2u;
        v36 = *(_DWORD *)(v34 + 56);
        if ( !v36 )
          break;
      }
      else
      {
        v36 = *(_DWORD *)(v34 + 56);
        if ( !v36 )
          break;
      }
    }
  }
  v38 = v17;
  v39 = (unsigned int)(*(_DWORD *)(v17 + 60) - 1);
  v40 = (_BYTE *)(*(_QWORD *)(v17 + 40) + 16 * v39);
  if ( *v40 == 111 )
  {
    v64 = v17;
    do
    {
      v41 = *(_QWORD *)(v38 + 48);
      v67 = v38;
      v68 = v39;
      if ( !v41 )
        break;
      while ( !(unsigned __int8)sub_1D18DA0(&v67, *(_QWORD *)(v41 + 16), v40, v33) )
      {
        v41 = *(_QWORD *)(v41 + 32);
        if ( !v41 )
          goto LABEL_43;
      }
      *(_DWORD *)(v38 + 28) = *(_DWORD *)(v35 + 192);
      v38 = *(_QWORD *)(v41 + 16);
      if ( *(__int16 *)(v38 + 24) < 0
        && (*(_BYTE *)(*(_QWORD *)(a1[2] + 8) + ((__int64)~*(__int16 *)(v38 + 24) << 6) + 8) & 0x10) != 0 )
      {
        *(_BYTE *)(v35 + 228) |= 2u;
      }
      v39 = (unsigned int)(*(_DWORD *)(v38 + 60) - 1);
      v40 = (_BYTE *)(*(_QWORD *)(v38 + 40) + 16 * v39);
    }
    while ( *v40 == 111 );
LABEL_43:
    v17 = v64;
  }
  if ( (*(_BYTE *)(v35 + 228) & 2) != 0 )
  {
    v60 = (unsigned int)v70;
    if ( (unsigned int)v70 >= HIDWORD(v70) )
    {
      sub_16CD150((__int64)&v69, v71, 0, 8, v31, v32);
      v60 = (unsigned int)v70;
    }
    *(_QWORD *)&v69[8 * v60] = v35;
    LODWORD(v70) = v70 + 1;
  }
  if ( *(_WORD *)(v17 + 24) == 2 )
    *(_BYTE *)(v35 + 229) |= 0x10u;
  v42 = *(_DWORD *)(v35 + 192);
  *(_QWORD *)v35 = v38;
  *(_DWORD *)(v38 + 28) = v42;
  sub_1D0E0F0((__int64)a1, v35);
  (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 64))(a1, v35);
  v16 = v79;
  if ( (_DWORD)v79 )
    goto LABEL_29;
LABEL_48:
  v43 = v69;
  v44 = v70;
LABEL_49:
  if ( v44 )
  {
LABEL_50:
    v45 = &v43[8 * v44];
    while ( 2 )
    {
      v46 = (__int64 *)*((_QWORD *)v45 - 1);
      LODWORD(v70) = --v44;
      v47 = *v46;
      while ( v47 )
      {
        if ( *(_WORD *)(v47 + 24) == 46 )
        {
          v48 = *(_QWORD *)(*(_QWORD *)(v47 + 32) + 80LL);
          v49 = *(_WORD *)(v48 + 24);
          v50 = (0x7FF0007FF22uLL >> v49) & 1;
          if ( v49 >= 0x2Bu )
            LOBYTE(v50) = 0;
          if ( v49 != 209 && !(_BYTE)v50 )
          {
            v51 = a1[6] + 272LL * *(int *)(v48 + 28);
            *(_BYTE *)(v51 + 228) |= 4u;
          }
        }
        v52 = *(_DWORD *)(v47 + 56);
        if ( !v52 )
          goto LABEL_49;
        v53 = (unsigned int *)(*(_QWORD *)(v47 + 32) + 40LL * (unsigned int)(v52 - 1));
        v47 = *(_QWORD *)v53;
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v53 + 40LL) + 16LL * v53[2]) != 111 )
        {
          if ( v44 )
            goto LABEL_50;
          goto LABEL_62;
        }
      }
      v45 -= 8;
      if ( v44 )
        continue;
      break;
    }
  }
LABEL_62:
  if ( v43 != v71 )
    _libc_free((unsigned __int64)v43);
  if ( v74 != v73 )
    _libc_free((unsigned __int64)v74);
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
}
