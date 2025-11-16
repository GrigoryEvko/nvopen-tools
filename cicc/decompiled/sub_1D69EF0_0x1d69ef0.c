// Function: sub_1D69EF0
// Address: 0x1d69ef0
//
__int64 __fastcall sub_1D69EF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // rsi
  unsigned __int8 v6; // al
  __int64 v7; // r15
  __int64 v8; // r14
  _QWORD *v9; // r13
  _QWORD *v10; // rax
  _QWORD *v11; // rbx
  char v12; // al
  __int64 v13; // rax
  unsigned int v14; // ecx
  _QWORD *v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  __int64 v20; // rdx
  unsigned __int8 v22; // al
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 *v27; // rbx
  __int64 v28; // rcx
  int v29; // esi
  __int64 v30; // rdi
  __int64 v31; // r11
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // rcx
  _QWORD *v35; // r14
  _QWORD *v36; // rax
  _QWORD *v37; // rbx
  unsigned int v38; // r15d
  unsigned __int8 v39; // al
  unsigned int v40; // edx
  __int64 v41; // r14
  bool v42; // r12
  unsigned int v43; // edx
  int v44; // r10d
  _QWORD *v45; // rbx
  int v46; // ecx
  __int64 v47; // rax
  __int64 v48; // r8
  bool v49; // zf
  __int64 *v50; // rsi
  int v51; // r15d
  __int64 *v52; // r9
  int v53; // eax
  __int64 v54; // rax
  __int64 v55; // r8
  __int64 *v56; // rsi
  __int64 v57; // rax
  __int64 v58; // r8
  __int64 v59; // rax
  int v60; // r11d
  __int64 *v61; // r10
  int v62; // edx
  int v63; // eax
  __int64 *v64; // [rsp+10h] [rbp-110h]
  __int64 v65; // [rsp+18h] [rbp-108h]
  __int64 v66; // [rsp+20h] [rbp-100h]
  unsigned __int8 v67; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v71; // [rsp+50h] [rbp-D0h]
  __int64 *v72; // [rsp+50h] [rbp-D0h]
  bool v73; // [rsp+5Fh] [rbp-C1h]
  __int64 v75; // [rsp+68h] [rbp-B8h]
  __int64 v76; // [rsp+78h] [rbp-A8h] BYREF
  __int64 v77; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v78; // [rsp+88h] [rbp-98h]
  __int64 *v79; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v80; // [rsp+98h] [rbp-88h]
  __int16 v81; // [rsp+A0h] [rbp-80h]
  __int64 v82; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v83; // [rsp+B8h] [rbp-68h]
  __int64 v84; // [rsp+C0h] [rbp-60h]
  unsigned int v85; // [rsp+C8h] [rbp-58h]
  __int64 v86; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v87; // [rsp+D8h] [rbp-48h]
  __int64 v88; // [rsp+E0h] [rbp-40h]
  unsigned int v89; // [rsp+E8h] [rbp-38h]

  v5 = *(__int64 **)a1;
  v75 = *(_QWORD *)(a1 + 40);
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v6 = sub_1D5D7E0(a4, v5, 0);
  v73 = 0;
  if ( v6 )
    v73 = *(_QWORD *)(a3 + 8LL * v6 + 120) != 0;
  v71 = 0;
  v7 = *(_QWORD *)(a1 + 8);
  if ( !v7 )
    goto LABEL_66;
  do
  {
    while ( 1 )
    {
      v9 = (_QWORD *)v7;
      v10 = sub_1648700(v7);
      v7 = *(_QWORD *)(v7 + 8);
      v11 = v10;
      v12 = *((_BYTE *)v10 + 16);
      if ( v12 == 77 )
        goto LABEL_8;
      if ( v12 != 60 )
      {
        if ( v12 != 50 )
          goto LABEL_8;
        if ( (*((_BYTE *)v11 + 23) & 0x40) != 0 )
        {
          v8 = *(_QWORD *)(*(v11 - 1) + 24LL);
          if ( *(_BYTE *)(v8 + 16) != 13 )
            goto LABEL_8;
        }
        else
        {
          v8 = v11[-3 * (*((_DWORD *)v11 + 5) & 0xFFFFFFF) + 3];
          if ( *(_BYTE *)(v8 + 16) != 13 )
            goto LABEL_8;
        }
        v78 = *(_DWORD *)(v8 + 32);
        if ( v78 > 0x40 )
          sub_16A4FD0((__int64)&v77, (const void **)(v8 + 24));
        else
          v77 = *(_QWORD *)(v8 + 24);
        sub_16A7490((__int64)&v77, 1);
        v40 = v78;
        v78 = 0;
        v80 = v40;
        v79 = (__int64 *)v77;
        if ( v40 > 0x40 )
        {
          sub_16A8890((__int64 *)&v79, (__int64 *)(v8 + 24));
          v43 = v80;
          v41 = (__int64)v79;
          v80 = 0;
          LODWORD(v87) = v43;
          v86 = (__int64)v79;
          if ( v43 > 0x40 )
          {
            v42 = v43 == (unsigned int)sub_16A57B0((__int64)&v86);
            if ( v41 )
            {
              j_j___libc_free_0_0(v41);
              if ( v80 > 0x40 )
              {
                if ( v79 )
                  j_j___libc_free_0_0(v79);
              }
            }
LABEL_53:
            if ( v78 > 0x40 && v77 )
              j_j___libc_free_0_0(v77);
            if ( !v42 )
              goto LABEL_8;
            goto LABEL_11;
          }
        }
        else
        {
          v41 = *(_QWORD *)(v8 + 24) & v77;
        }
        v42 = v41 == 0;
        goto LABEL_53;
      }
LABEL_11:
      v13 = v11[5];
      v76 = v13;
      if ( v75 == v13 )
      {
        v67 = v73 && *((_BYTE *)v11 + 16) == 60;
        if ( !v67 )
          goto LABEL_8;
        v22 = sub_1D5D7E0(a4, (__int64 *)*v11, 0);
        if ( v22 )
        {
          if ( *(_QWORD *)(a3 + 8LL * v22 + 120) )
            goto LABEL_8;
        }
        v23 = v11[5];
        v86 = 0;
        v87 = 0;
        v88 = 0;
        v89 = 0;
        v66 = v23;
        if ( *((_BYTE *)v11 + 16) != 60 )
          BUG();
        v24 = v11[1];
        v71 = 0;
        if ( !v24 )
        {
LABEL_65:
          j___libc_free_0(v24);
          goto LABEL_8;
        }
        v64 = v11;
        v65 = v7;
        while ( 1 )
        {
          v35 = (_QWORD *)v24;
          v36 = sub_1648700(v24);
          v24 = *(_QWORD *)(v24 + 8);
          v37 = v36;
          v38 = sub_1F43D70(a3, (unsigned int)*((unsigned __int8 *)v36 + 16) - 24);
          if ( !v38 )
            goto LABEL_40;
          v39 = sub_1D5D7E0(a4, (__int64 *)*v37, 1u);
          if ( (v39 == 1 || v39 && *(_QWORD *)(a3 + 8LL * v39 + 120))
            && (v38 > 0x102 || (*(_BYTE *)(v38 + a3 + 259LL * v39 + 2422) & 0xFB) == 0) )
          {
            goto LABEL_40;
          }
          if ( *((_BYTE *)v37 + 16) == 77 )
            goto LABEL_40;
          v25 = v37[5];
          v77 = v25;
          if ( v66 == v25 )
            goto LABEL_40;
          if ( !v85 )
            break;
          LODWORD(v26) = (v85 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v27 = (__int64 *)(v83 + 16LL * (unsigned int)v26);
          v28 = *v27;
          if ( v25 != *v27 )
          {
            v60 = 1;
            v61 = 0;
            while ( v28 != -8 )
            {
              if ( v28 == -16 && !v61 )
                v61 = v27;
              v26 = (v85 - 1) & ((_DWORD)v26 + v60);
              v27 = (__int64 *)(v83 + 16 * v26);
              v28 = *v27;
              if ( v25 == *v27 )
                goto LABEL_36;
              ++v60;
            }
            if ( v61 )
              v27 = v61;
            ++v82;
            v62 = v84 + 1;
            if ( 4 * ((int)v84 + 1) < 3 * v85 )
            {
              if ( v85 - HIDWORD(v84) - v62 <= v85 >> 3 )
              {
                sub_1D69D30((__int64)&v82, v85);
                sub_1D67DD0((__int64)&v82, &v77, &v79);
                v27 = v79;
                v25 = v77;
                v62 = v84 + 1;
              }
              goto LABEL_117;
            }
LABEL_121:
            sub_1D69D30((__int64)&v82, 2 * v85);
            sub_1D67DD0((__int64)&v82, &v77, &v79);
            v27 = v79;
            v25 = v77;
            v62 = v84 + 1;
LABEL_117:
            LODWORD(v84) = v62;
            if ( *v27 != -8 )
              --HIDWORD(v84);
            *v27 = v25;
            v27[1] = 0;
          }
LABEL_36:
          v29 = v89;
          if ( !v89 )
          {
            ++v86;
            goto LABEL_108;
          }
          v30 = v77;
          v31 = v77;
          v32 = (v89 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
          v33 = (__int64 *)(v87 + 16LL * v32);
          v34 = *v33;
          if ( v77 != *v33 )
          {
            v51 = 1;
            v52 = 0;
            while ( v34 != -8 )
            {
              if ( !v52 && v34 == -16 )
                v52 = v33;
              v63 = v51++;
              v32 = (v89 - 1) & (v63 + v32);
              v33 = (__int64 *)(v87 + 16LL * v32);
              v34 = *v33;
              if ( v77 == *v33 )
                goto LABEL_38;
            }
            if ( !v52 )
              v52 = v33;
            ++v86;
            v53 = v88 + 1;
            if ( 4 * ((int)v88 + 1) < 3 * v89 )
            {
              if ( v89 - HIDWORD(v88) - v53 > v89 >> 3 )
                goto LABEL_97;
              goto LABEL_109;
            }
LABEL_108:
            v29 = 2 * v89;
LABEL_109:
            sub_1D69170((__int64)&v86, v29);
            sub_1D67D20((__int64)&v86, &v77, &v79);
            v52 = v79;
            v31 = v77;
            v53 = v88 + 1;
LABEL_97:
            LODWORD(v88) = v53;
            if ( *v52 != -8 )
              --HIDWORD(v88);
            *v52 = v31;
            v52[1] = 0;
            v30 = v77;
            if ( v27[1] )
              goto LABEL_40;
            goto LABEL_100;
          }
LABEL_38:
          if ( !v27[1] && !v33[1] )
          {
            v52 = v33;
LABEL_100:
            v72 = v52;
            v54 = sub_157EE30(v30);
            v55 = v54;
            if ( v54 )
              v55 = v54 - 24;
            v81 = 257;
            v56 = *(__int64 **)(a1 - 48);
            if ( *(_BYTE *)(a1 + 16) == 49 )
              v57 = sub_15FB440(25, v56, a2, (__int64)&v79, v55);
            else
              v57 = sub_15FB440(24, v56, a2, (__int64)&v79, v55);
            v27[1] = v57;
            v58 = *(_QWORD *)(sub_157EE30(v77) + 8);
            v81 = 257;
            if ( v58 )
              v58 -= 24;
            v59 = sub_15FDBD0((unsigned int)*((unsigned __int8 *)v64 + 16) - 24, v27[1], *v64, (__int64)&v79, v58);
            v72[1] = v59;
            sub_1593B40(v35, v59);
            v71 = v67;
          }
LABEL_40:
          if ( !v24 )
          {
            v7 = v65;
            v24 = v87;
            goto LABEL_65;
          }
        }
        ++v82;
        goto LABEL_121;
      }
      if ( !v85 )
      {
        ++v82;
        goto LABEL_85;
      }
      v14 = (v85 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v15 = (_QWORD *)(v83 + 16LL * v14);
      v16 = *v15;
      if ( *v15 == v13 )
        break;
      v44 = 1;
      v45 = 0;
      while ( v16 != -8 )
      {
        if ( v16 == -16 && !v45 )
          v45 = v15;
        v14 = (v85 - 1) & (v44 + v14);
        v15 = (_QWORD *)(v83 + 16LL * v14);
        v16 = *v15;
        if ( v13 == *v15 )
          goto LABEL_14;
        ++v44;
      }
      if ( !v45 )
        v45 = v15;
      ++v82;
      v46 = v84 + 1;
      if ( 4 * ((int)v84 + 1) < 3 * v85 )
      {
        if ( v85 - HIDWORD(v84) - v46 <= v85 >> 3 )
        {
          sub_1D69D30((__int64)&v82, v85);
          sub_1D67DD0((__int64)&v82, &v76, &v86);
          v45 = (_QWORD *)v86;
          v13 = v76;
          v46 = v84 + 1;
        }
        goto LABEL_74;
      }
LABEL_85:
      sub_1D69D30((__int64)&v82, 2 * v85);
      sub_1D67DD0((__int64)&v82, &v76, &v86);
      v45 = (_QWORD *)v86;
      v13 = v76;
      v46 = v84 + 1;
LABEL_74:
      LODWORD(v84) = v46;
      if ( *v45 != -8 )
        --HIDWORD(v84);
      *v45 = v13;
      v45[1] = 0;
LABEL_77:
      v47 = sub_157EE30(v76);
      v48 = v47;
      if ( v47 )
        v48 = v47 - 24;
      v49 = *(_BYTE *)(a1 + 16) == 49;
      LOWORD(v88) = 257;
      v50 = *(__int64 **)(a1 - 48);
      if ( v49 )
        v45[1] = sub_15FB440(25, v50, a2, (__int64)&v86, v48);
      else
        v45[1] = sub_15FB440(24, v50, a2, (__int64)&v86, v48);
      v17 = v45[1];
      v71 = 1;
      if ( *v9 )
      {
LABEL_16:
        v18 = v9[1];
        v19 = v9[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v19 = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
      }
      *v9 = v17;
      if ( v17 )
        goto LABEL_19;
LABEL_8:
      if ( !v7 )
        goto LABEL_22;
    }
LABEL_14:
    v17 = v15[1];
    if ( !v17 )
    {
      v45 = v15;
      goto LABEL_77;
    }
    if ( *v9 )
      goto LABEL_16;
    *v9 = v17;
LABEL_19:
    v20 = *(_QWORD *)(v17 + 8);
    v9[1] = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 16) = (unsigned __int64)(v9 + 1) | *(_QWORD *)(v20 + 16) & 3LL;
    v9[2] = (v17 + 8) | v9[2] & 3LL;
    *(_QWORD *)(v17 + 8) = v9;
  }
  while ( v7 );
LABEL_22:
  if ( !*(_QWORD *)(a1 + 8) )
LABEL_66:
    sub_15F20C0((_QWORD *)a1);
  j___libc_free_0(v83);
  return v71;
}
