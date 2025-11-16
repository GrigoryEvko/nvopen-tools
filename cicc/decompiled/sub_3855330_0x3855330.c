// Function: sub_3855330
// Address: 0x3855330
//
__int64 __fastcall sub_3855330(__int64 a1, __int64 a2)
{
  unsigned int v4; // eax
  __int64 v5; // rdx
  bool v6; // zf
  unsigned int v7; // eax
  __int64 v8; // r13
  __int64 v9; // r11
  __int64 v10; // rcx
  __int64 v11; // r14
  int v12; // edx
  int v13; // ecx
  __int64 v14; // rdi
  unsigned int v15; // edx
  __int64 v16; // rsi
  char v17; // al
  int v18; // r8d
  unsigned int v19; // esi
  __int64 v20; // r9
  unsigned int v21; // r8d
  _QWORD *v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // r14
  bool v27; // dl
  __int64 v28; // r15
  __int64 v30; // rdx
  __int64 *v31; // rax
  bool v32; // cc
  int v33; // edx
  _QWORD *v34; // rax
  int v35; // eax
  int v36; // esi
  __int64 v37; // rcx
  unsigned int v38; // edx
  __int64 *v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rdx
  bool v42; // al
  int v43; // eax
  int v44; // esi
  __int64 v45; // rcx
  unsigned int v46; // edx
  __int64 *v47; // rax
  __int64 v48; // rdi
  _QWORD *v49; // rcx
  int v50; // eax
  int v51; // eax
  int v52; // esi
  int v53; // esi
  __int64 v54; // r8
  unsigned int v55; // edx
  __int64 v56; // rdi
  int v57; // r15d
  _QWORD *v58; // r9
  int v59; // edx
  int v60; // edx
  __int64 v61; // rdi
  _QWORD *v62; // r8
  unsigned int v63; // r15d
  int v64; // r10d
  __int64 v65; // rsi
  int v66; // eax
  int v67; // r8d
  int v68; // eax
  int v69; // r8d
  unsigned __int64 v70; // rdi
  unsigned int v71; // edx
  __int64 v72; // rax
  __int64 v73; // [rsp+0h] [rbp-E0h]
  bool v74; // [rsp+8h] [rbp-D8h]
  __int64 v75; // [rsp+8h] [rbp-D8h]
  __int64 v76; // [rsp+8h] [rbp-D8h]
  __int64 v77; // [rsp+8h] [rbp-D8h]
  int v78; // [rsp+8h] [rbp-D8h]
  __int64 v79; // [rsp+8h] [rbp-D8h]
  __int64 v80; // [rsp+8h] [rbp-D8h]
  __int64 v81; // [rsp+8h] [rbp-D8h]
  __int64 v82; // [rsp+10h] [rbp-D0h]
  bool v83; // [rsp+1Fh] [rbp-C1h]
  __int64 v84; // [rsp+20h] [rbp-C0h]
  __int64 v85; // [rsp+28h] [rbp-B8h]
  __int64 v86; // [rsp+28h] [rbp-B8h]
  __int64 v87; // [rsp+38h] [rbp-A8h] BYREF
  const void *v88; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v89; // [rsp+48h] [rbp-98h]
  __int64 v90; // [rsp+50h] [rbp-90h]
  const void *v91; // [rsp+58h] [rbp-88h] BYREF
  unsigned int v92; // [rsp+60h] [rbp-80h]
  __int64 v93; // [rsp+70h] [rbp-70h] BYREF
  unsigned __int64 v94; // [rsp+78h] [rbp-68h] BYREF
  unsigned int v95; // [rsp+80h] [rbp-60h]
  __int64 v96; // [rsp+90h] [rbp-50h] BYREF
  __int64 v97; // [rsp+98h] [rbp-48h] BYREF
  __int64 v98; // [rsp+A0h] [rbp-40h]
  __int64 v99; // [rsp+A8h] [rbp-38h]

  v4 = 8 * sub_15A9520(*(_QWORD *)(a1 + 40), 0);
  v89 = v4;
  if ( v4 <= 0x40 )
  {
    v88 = 0;
    v5 = *(_QWORD *)a2;
    v90 = 0;
    v6 = *(_BYTE *)(v5 + 8) == 15;
    v92 = v4;
    v83 = v6;
    goto LABEL_3;
  }
  sub_16A4EF0((__int64)&v88, 0, 0);
  v30 = *(_QWORD *)a2;
  v90 = 0;
  v6 = *(_BYTE *)(v30 + 8) == 15;
  v92 = v89;
  v83 = v6;
  if ( v89 <= 0x40 )
  {
LABEL_3:
    v91 = v88;
    v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( !v7 )
      goto LABEL_31;
    goto LABEL_4;
  }
  sub_16A4FD0((__int64)&v91, &v88);
  v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v7 )
  {
LABEL_4:
    v84 = 0;
    v8 = 0;
    v9 = 8LL * v7;
    v85 = 0;
    v82 = a1 + 320;
    while ( 1 )
    {
      v17 = *(_BYTE *)(a2 + 23) & 0x40;
      if ( v17 )
        v10 = *(_QWORD *)(a2 - 8);
      else
        v10 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v11 = *(_QWORD *)(v8 + v10 + 24LL * *(unsigned int *)(a2 + 56) + 8);
      v12 = *(_DWORD *)(a1 + 288);
      if ( v12 )
      {
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 272);
        v15 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v16 = *(_QWORD *)(v14 + 8LL * v15);
        if ( v11 == v16 )
          goto LABEL_8;
        v18 = 1;
        while ( v16 != -8 )
        {
          v15 = v13 & (v18 + v15);
          v16 = *(_QWORD *)(v14 + 8LL * v15);
          if ( v11 == v16 )
            goto LABEL_8;
          ++v18;
        }
      }
      v19 = *(_DWORD *)(a1 + 344);
      if ( !v19 )
        break;
      v20 = *(_QWORD *)(a1 + 328);
      v21 = (v19 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v22 = (_QWORD *)(v20 + 16LL * v21);
      v23 = *v22;
      if ( v11 != *v22 )
      {
        v78 = 1;
        v49 = 0;
        while ( v23 != -8 )
        {
          if ( v23 == -16 && !v49 )
            v49 = v22;
          v21 = (v19 - 1) & (v78 + v21);
          v22 = (_QWORD *)(v20 + 16LL * v21);
          v23 = *v22;
          if ( v11 == *v22 )
            goto LABEL_15;
          ++v78;
        }
        v50 = *(_DWORD *)(a1 + 336);
        if ( !v49 )
          v49 = v22;
        ++*(_QWORD *)(a1 + 320);
        v51 = v50 + 1;
        if ( 4 * v51 < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a1 + 340) - v51 <= v19 >> 3 )
          {
            v80 = v9;
            sub_1447B20(v82, v19);
            v59 = *(_DWORD *)(a1 + 344);
            if ( !v59 )
            {
LABEL_139:
              ++*(_DWORD *)(a1 + 336);
              BUG();
            }
            v60 = v59 - 1;
            v61 = *(_QWORD *)(a1 + 328);
            v62 = 0;
            v63 = v60 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v9 = v80;
            v64 = 1;
            v51 = *(_DWORD *)(a1 + 336) + 1;
            v49 = (_QWORD *)(v61 + 16LL * v63);
            v65 = *v49;
            if ( v11 != *v49 )
            {
              while ( v65 != -8 )
              {
                if ( !v62 && v65 == -16 )
                  v62 = v49;
                v63 = v60 & (v64 + v63);
                v49 = (_QWORD *)(v61 + 16LL * v63);
                v65 = *v49;
                if ( v11 == *v49 )
                  goto LABEL_78;
                ++v64;
              }
              if ( v62 )
                v49 = v62;
            }
          }
          goto LABEL_78;
        }
LABEL_83:
        v79 = v9;
        sub_1447B20(v82, 2 * v19);
        v52 = *(_DWORD *)(a1 + 344);
        if ( !v52 )
          goto LABEL_139;
        v53 = v52 - 1;
        v9 = v79;
        v54 = *(_QWORD *)(a1 + 328);
        v55 = v53 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v51 = *(_DWORD *)(a1 + 336) + 1;
        v49 = (_QWORD *)(v54 + 16LL * v55);
        v56 = *v49;
        if ( v11 != *v49 )
        {
          v57 = 1;
          v58 = 0;
          while ( v56 != -8 )
          {
            if ( v56 == -16 && !v58 )
              v58 = v49;
            v55 = v53 & (v57 + v55);
            v49 = (_QWORD *)(v54 + 16LL * v55);
            v56 = *v49;
            if ( v11 == *v49 )
              goto LABEL_78;
            ++v57;
          }
          if ( v58 )
            v49 = v58;
        }
LABEL_78:
        *(_DWORD *)(a1 + 336) = v51;
        if ( *v49 != -8 )
          --*(_DWORD *)(a1 + 340);
        *v49 = v11;
        v49[1] = 0;
        v17 = *(_BYTE *)(a2 + 23) & 0x40;
        goto LABEL_17;
      }
LABEL_15:
      v24 = v22[1];
      if ( v24 && v24 != *(_QWORD *)(a2 + 40) )
        goto LABEL_8;
LABEL_17:
      if ( v17 )
        v25 = *(_QWORD *)(a2 - 8);
      else
        v25 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v26 = *(_QWORD *)(v25 + 3 * v8);
      if ( !v26 )
        BUG();
      if ( v26 == a2 )
        goto LABEL_8;
      v27 = 0;
      v28 = v26;
      if ( *(_BYTE *)(v26 + 16) > 0x10u )
      {
        v43 = *(_DWORD *)(a1 + 160);
        if ( v43 )
        {
          v44 = v43 - 1;
          v45 = *(_QWORD *)(a1 + 144);
          v46 = (v43 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v47 = (__int64 *)(v45 + 16LL * v46);
          v48 = *v47;
          if ( *v47 == v26 )
          {
LABEL_64:
            v28 = v47[1];
            v27 = v83 && v28 == 0;
            goto LABEL_22;
          }
          v68 = 1;
          while ( v48 != -8 )
          {
            v69 = v68 + 1;
            v46 = v44 & (v68 + v46);
            v47 = (__int64 *)(v45 + 16LL * v46);
            v48 = *v47;
            if ( v26 == *v47 )
              goto LABEL_64;
            v68 = v69;
          }
        }
        v27 = v83;
        v28 = 0;
      }
LABEL_22:
      v93 = 0;
      v95 = v89;
      if ( v89 > 0x40 )
      {
        v73 = v9;
        v74 = v27;
        sub_16A4FD0((__int64)&v94, &v88);
        v9 = v73;
        v27 = v74;
      }
      else
      {
        v94 = (unsigned __int64)v88;
      }
      if ( v27 )
      {
        v35 = *(_DWORD *)(a1 + 256);
        if ( v35 )
        {
          v36 = v35 - 1;
          v37 = *(_QWORD *)(a1 + 240);
          v38 = (v35 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v39 = (__int64 *)(v37 + 32LL * v38);
          v40 = *v39;
          if ( *v39 == v26 )
          {
LABEL_52:
            v41 = v39[1];
            v96 = v41;
            LODWORD(v98) = *((_DWORD *)v39 + 6);
            if ( (unsigned int)v98 > 0x40 )
            {
              v81 = v9;
              sub_16A4FD0((__int64)&v97, (const void **)v39 + 2);
              v41 = v96;
              v9 = v81;
            }
            else
            {
              v97 = v39[2];
            }
LABEL_66:
            v93 = v41;
            if ( v95 > 0x40 && v94 )
            {
              v77 = v9;
              j_j___libc_free_0_0(v94);
              v41 = v93;
              v9 = v77;
            }
            v94 = v97;
            v95 = v98;
LABEL_70:
            if ( !v41 )
              goto LABEL_28;
            goto LABEL_26;
          }
          v66 = 1;
          while ( v40 != -8 )
          {
            v67 = v66 + 1;
            v38 = v36 & (v66 + v38);
            v39 = (__int64 *)(v37 + 32LL * v38);
            v40 = *v39;
            if ( v26 == *v39 )
              goto LABEL_52;
            v66 = v67;
          }
        }
        v96 = 0;
        v41 = 0;
        LODWORD(v98) = 1;
        v97 = 0;
        goto LABEL_66;
      }
      if ( !v28 )
      {
        v41 = v93;
        goto LABEL_70;
      }
LABEL_26:
      if ( v85 )
      {
        if ( v28 != v85 )
          goto LABEL_28;
      }
      else if ( v84 )
      {
        if ( v90 != v93 )
          goto LABEL_28;
        if ( v92 <= 0x40 )
        {
          if ( v91 != (const void *)v94 )
          {
LABEL_28:
            if ( v95 > 0x40 && v94 )
              j_j___libc_free_0_0(v94);
            goto LABEL_31;
          }
        }
        else
        {
          v75 = v9;
          v42 = sub_16A5220((__int64)&v91, (const void **)&v94);
          v9 = v75;
          if ( !v42 )
            goto LABEL_28;
        }
      }
      else
      {
        v85 = v28;
        if ( !v28 )
        {
          v90 = v93;
          if ( v92 <= 0x40 && v95 <= 0x40 )
          {
            v92 = v95;
            v91 = (const void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v95) & v94);
          }
          else
          {
            v86 = v9;
            sub_16A51C0((__int64)&v91, (__int64)&v94);
            v9 = v86;
            if ( v95 > 0x40 && v94 )
            {
              j_j___libc_free_0_0(v94);
              v9 = v86;
            }
          }
          v84 = v26;
          v85 = 0;
          goto LABEL_8;
        }
      }
      if ( v95 > 0x40 && v94 )
      {
        v76 = v9;
        j_j___libc_free_0_0(v94);
        v9 = v76;
      }
LABEL_8:
      v8 += 8;
      if ( v8 == v9 )
      {
        if ( !v85 )
          goto LABEL_42;
        v96 = a2;
        sub_38526A0(a1 + 136, &v96)[1] = v85;
        goto LABEL_31;
      }
    }
    ++*(_QWORD *)(a1 + 320);
    goto LABEL_83;
  }
  v84 = 0;
LABEL_42:
  if ( v90 )
  {
    v96 = a2;
    v31 = sub_3854530(a1 + 232, &v96);
    v32 = *((_DWORD *)v31 + 6) <= 0x40u;
    v31[1] = v90;
    if ( v32 && v92 <= 0x40 )
    {
      v70 = (unsigned __int64)v91;
      v31[2] = (__int64)v91;
      v71 = v92;
      *((_DWORD *)v31 + 6) = v92;
      if ( v71 > 0x40 )
      {
        v72 = (unsigned int)(((unsigned __int64)v71 + 63) >> 6) - 1;
        *(_QWORD *)(v70 + 8 * v72) &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v71;
      }
      else
      {
        v31[2] = v70 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v71);
      }
    }
    else
    {
      sub_16A51C0((__int64)(v31 + 2), (__int64)&v91);
    }
    v33 = *(_DWORD *)(a1 + 184);
    v96 = 0;
    v97 = -1;
    v98 = 0;
    v99 = 0;
    if ( v33 )
    {
      if ( *(_DWORD *)(a1 + 216) && sub_384F1D0(a1, v84, &v87, &v96) )
      {
        v93 = a2;
        v34 = sub_176FB00(a1 + 168, &v93);
        v34[1] = v87;
      }
    }
  }
LABEL_31:
  if ( v92 > 0x40 && v91 )
    j_j___libc_free_0_0((unsigned __int64)v91);
  if ( v89 > 0x40 && v88 )
    j_j___libc_free_0_0((unsigned __int64)v88);
  return 1;
}
