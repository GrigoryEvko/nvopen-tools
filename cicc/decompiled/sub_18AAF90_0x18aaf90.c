// Function: sub_18AAF90
// Address: 0x18aaf90
//
__int64 __fastcall sub_18AAF90(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v5; // rsi
  __int64 v6; // r13
  char v7; // al
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned int v10; // r12d
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // r9
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned int *v21; // r11
  unsigned int *v22; // r10
  int v23; // eax
  __int64 *v24; // rdi
  unsigned __int64 v25; // rax
  unsigned int v26; // esi
  __int64 v27; // r12
  __int64 v28; // rdi
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // rcx
  __int64 *v32; // rax
  __int64 v34; // rax
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // rax
  __int64 *v37; // rax
  __int64 v38; // rax
  __int64 v39; // r10
  int v40; // eax
  int v41; // edi
  unsigned int *v42; // rax
  __int64 v43; // rax
  unsigned int *v44; // rdx
  _BOOL8 v45; // rdi
  unsigned __int64 v46; // rcx
  __int64 *v47; // rsi
  unsigned int v48; // edi
  __int64 *v49; // rcx
  unsigned int *v50; // rdi
  int v51; // r10d
  __int64 *v52; // r9
  int v53; // ecx
  int v54; // ecx
  __int64 v55; // rdi
  __int64 v56; // [rsp+0h] [rbp-F0h]
  unsigned int *v57; // [rsp+8h] [rbp-E8h]
  __int64 v58; // [rsp+10h] [rbp-E0h]
  __int64 v59; // [rsp+18h] [rbp-D8h]
  int v60; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v61; // [rsp+20h] [rbp-D0h]
  __int64 v62; // [rsp+30h] [rbp-C0h]
  unsigned int v63; // [rsp+38h] [rbp-B8h]
  int v64; // [rsp+38h] [rbp-B8h]
  unsigned int v65; // [rsp+38h] [rbp-B8h]
  unsigned int *v66; // [rsp+38h] [rbp-B8h]
  unsigned int *v67; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v68; // [rsp+40h] [rbp-B0h]
  __int64 v69; // [rsp+48h] [rbp-A8h]
  __int64 v70; // [rsp+50h] [rbp-A0h]
  unsigned int v71; // [rsp+58h] [rbp-98h]
  unsigned __int8 v72; // [rsp+5Eh] [rbp-92h]
  char v73; // [rsp+5Fh] [rbp-91h]
  unsigned int v74; // [rsp+68h] [rbp-88h] BYREF
  unsigned int v75; // [rsp+6Ch] [rbp-84h] BYREF
  __int64 v76; // [rsp+70h] [rbp-80h] BYREF
  __int64 v77; // [rsp+78h] [rbp-78h] BYREF
  unsigned __int64 v78[2]; // [rsp+80h] [rbp-70h] BYREF
  char v79; // [rsp+90h] [rbp-60h]
  __int64 v80; // [rsp+A0h] [rbp-50h]
  unsigned __int64 *v81; // [rsp+A8h] [rbp-48h]
  unsigned int *v82; // [rsp+B0h] [rbp-40h]
  unsigned int *v83; // [rsp+B8h] [rbp-38h]

  v62 = a2 + 72;
  if ( *(_QWORD *)(a2 + 80) == a2 + 72 )
    return 0;
  v72 = 0;
  v56 = a1 + 64;
  v70 = *(_QWORD *)(a2 + 80);
  do
  {
    if ( !v70 )
      BUG();
    v3 = *(_QWORD *)(v70 + 24);
    v4 = v70 + 16;
    if ( v3 == v70 + 16 )
      goto LABEL_42;
    v68 = 0;
    v73 = 0;
    do
    {
      while ( 1 )
      {
        if ( !v3 )
          BUG();
        if ( !*(_QWORD *)(v3 + 24) )
          goto LABEL_35;
        v5 = sub_15C70A0(v3 + 24);
        v6 = v5 ? sub_393D1F0(*(_QWORD *)(a1 + 1200), v5) : *(_QWORD *)(a1 + 1200);
        if ( !v6 )
          goto LABEL_35;
        v7 = *(_BYTE *)(v3 - 8);
        if ( v7 == 26 )
          goto LABEL_35;
        v69 = v3 - 24;
        if ( v7 == 78 )
        {
          v34 = *(_QWORD *)(v3 - 48);
          if ( !*(_BYTE *)(v34 + 16) && (*(_BYTE *)(v34 + 33) & 0x20) != 0 )
            goto LABEL_35;
          v35 = v69 | 4;
        }
        else
        {
          if ( v7 != 29 )
            break;
          v35 = v69 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v36 = v35 & 0xFFFFFFFFFFFFFFF8LL;
        v37 = (__int64 *)((v35 & 4) != 0 ? v36 - 24 : v36 - 72);
        v38 = *v37;
        if ( v38 )
        {
          if ( *(_BYTE *)(v38 + 16) > 0x10u )
          {
            v46 = v35 & 0xFFFFFFFFFFFFFFF8LL;
            if ( *(_BYTE *)(v46 + 16) != 78 || *(_BYTE *)(*(_QWORD *)(v46 - 24) + 16LL) != 20 )
              break;
          }
        }
        if ( !sub_18A8560(a1, v69) )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        v73 = 1;
        if ( v4 == v3 )
          goto LABEL_36;
      }
      v8 = sub_15C70A0(v3 + 24);
      v74 = sub_393D1C0(v8);
      v9 = *(_QWORD *)(v8 - 8LL * *(unsigned int *)(v8 + 8));
      v10 = 0;
      if ( *(_BYTE *)v9 == 19 )
      {
        v11 = *(_DWORD *)(v9 + 24);
        v10 = 0;
        if ( (v11 & 1) == 0 )
        {
          v10 = (v11 >> 1) & 0x1F;
          if ( ((v11 >> 1) & 0x20) != 0 )
            v10 |= (v11 >> 2) & 0xFE0;
        }
      }
      v75 = v10;
      v78[0] = __PAIR64__(v10, v74);
      v63 = v74;
      v12 = sub_18A8380(v6 + 32, (unsigned int *)v78);
      if ( v12 == v6 + 40 )
        goto LABEL_35;
      v13 = *(unsigned int *)(a1 + 1176);
      v14 = *(_QWORD *)(v12 + 40);
      v76 = v6;
      v79 &= ~1u;
      v15 = v63;
      v61 = v14;
      v78[0] = v14;
      if ( !(_DWORD)v13 )
      {
        ++*(_QWORD *)(a1 + 1152);
        goto LABEL_106;
      }
      v16 = *(_QWORD *)(a1 + 1160);
      v17 = ((_DWORD)v13 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v18 = v16 + 56 * v17;
      v19 = *(_QWORD *)v18;
      if ( v6 == *(_QWORD *)v18 )
        goto LABEL_19;
      v64 = 1;
      v39 = 0;
      while ( 1 )
      {
        if ( v19 == -8 )
        {
          v40 = *(_DWORD *)(a1 + 1168);
          if ( v39 )
            v18 = v39;
          ++*(_QWORD *)(a1 + 1152);
          v41 = v40 + 1;
          if ( 4 * (v40 + 1) < (unsigned int)(3 * v13) )
          {
            if ( (int)v13 - *(_DWORD *)(a1 + 1172) - v41 > (unsigned int)v13 >> 3 )
            {
LABEL_67:
              *(_DWORD *)(a1 + 1168) = v41;
              if ( *(_QWORD *)v18 != -8 )
                --*(_DWORD *)(a1 + 1172);
              v22 = (unsigned int *)(v18 + 16);
              *(_QWORD *)v18 = v6;
              *(_DWORD *)(v18 + 16) = 0;
              v21 = (unsigned int *)(v18 + 16);
              *(_QWORD *)(v18 + 24) = 0;
              *(_QWORD *)(v18 + 32) = v18 + 16;
              *(_QWORD *)(v18 + 40) = v18 + 16;
              *(_QWORD *)(v18 + 48) = 0;
              goto LABEL_70;
            }
            v60 = v15;
LABEL_107:
            sub_18AA9E0(a1 + 1152, v13);
            sub_18A87C0(a1 + 1152, &v76, &v77);
            v18 = v77;
            v6 = v76;
            LODWORD(v15) = v60;
            v41 = *(_DWORD *)(a1 + 1168) + 1;
            goto LABEL_67;
          }
LABEL_106:
          v60 = v15;
          LODWORD(v13) = 2 * v13;
          goto LABEL_107;
        }
        if ( v39 || v19 != -16 )
          v18 = v39;
        v17 = ((_DWORD)v13 - 1) & (unsigned int)(v64 + v17);
        ++v64;
        v19 = *(_QWORD *)(v16 + 56LL * (unsigned int)v17);
        if ( v6 == v19 )
          break;
        v39 = v18;
        v18 = v16 + 56LL * (unsigned int)v17;
      }
      v18 = v16 + 56LL * (unsigned int)v17;
LABEL_19:
      v20 = *(_QWORD *)(v18 + 24);
      v21 = (unsigned int *)(v18 + 16);
      v22 = (unsigned int *)(v18 + 16);
      if ( !v20 )
      {
LABEL_70:
        v59 = (__int64)v22;
        v58 = v18;
        v65 = v15;
        v57 = v21;
        v42 = (unsigned int *)sub_22077B0(48);
        v42[9] = v10;
        v42[8] = v65;
        v42[10] = 0;
        v71 = v65;
        v66 = v42;
        v43 = sub_18A9FE0((_QWORD *)(v58 + 8), v59, v42 + 8);
        if ( v44 )
        {
          v45 = 1;
          if ( !v43 && v57 != v44 && v71 >= v44[8] )
          {
            v45 = 0;
            if ( v71 == v44[8] )
              v45 = v10 < v44[9];
          }
          v13 = (__int64)v66;
          sub_220F040(v45, v66, v44, v57);
          v18 = v58;
          v22 = v66;
          ++*(_QWORD *)(v58 + 48);
        }
        else
        {
          v50 = v66;
          v13 = 48;
          v67 = (unsigned int *)v43;
          j_j___libc_free_0(v50, 48);
          v22 = v67;
        }
        goto LABEL_29;
      }
      while ( 1 )
      {
LABEL_23:
        if ( (unsigned int)v15 > *(_DWORD *)(v20 + 32) )
        {
          v20 = *(_QWORD *)(v20 + 24);
          goto LABEL_25;
        }
        if ( (_DWORD)v15 == *(_DWORD *)(v20 + 32) && v10 > *(_DWORD *)(v20 + 36) )
          break;
        v22 = (unsigned int *)v20;
        v20 = *(_QWORD *)(v20 + 16);
        if ( !v20 )
          goto LABEL_26;
      }
      v20 = *(_QWORD *)(v20 + 24);
LABEL_25:
      if ( v20 )
        goto LABEL_23;
LABEL_26:
      if ( v21 == v22 || (unsigned int)v15 < v22[8] || (_DWORD)v15 == v22[8] && v10 < v22[9] )
        goto LABEL_70;
LABEL_29:
      v23 = v22[10] + 1;
      v22[10] = v23;
      if ( v23 == 1 )
      {
        *(_QWORD *)(a1 + 1184) += v61;
        v24 = *(__int64 **)(a1 + 1264);
        v80 = v3 - 24;
        v81 = v78;
        v82 = &v74;
        v83 = &v75;
        sub_18A4B90(v24, v13, v17, v15, v16, v18, v69, v78, &v74, &v75);
      }
      if ( (v79 & 1) == 0 )
      {
        v25 = v78[0];
        v73 = 1;
        if ( v68 >= v78[0] )
          v25 = v68;
        v68 = v25;
      }
LABEL_35:
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v4 != v3 );
LABEL_36:
    if ( !v73 )
      goto LABEL_42;
    v26 = *(_DWORD *)(a1 + 24);
    v27 = v70 - 24;
    v77 = v70 - 24;
    if ( !v26 )
    {
      ++*(_QWORD *)a1;
LABEL_103:
      v26 *= 2;
LABEL_104:
      sub_18AACA0(a1, v26);
      sub_18AA140(a1, &v77, v78);
      v30 = (__int64 *)v78[0];
      v55 = v77;
      v54 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_99;
    }
    v28 = *(_QWORD *)(a1 + 8);
    v29 = (v26 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
    v30 = (__int64 *)(v28 + 16LL * v29);
    v31 = *v30;
    if ( v27 == *v30 )
      goto LABEL_39;
    v51 = 1;
    v52 = 0;
    while ( v31 != -8 )
    {
      if ( v31 == -16 && !v52 )
        v52 = v30;
      v29 = (v26 - 1) & (v51 + v29);
      v30 = (__int64 *)(v28 + 16LL * v29);
      v31 = *v30;
      if ( v27 == *v30 )
        goto LABEL_39;
      ++v51;
    }
    v53 = *(_DWORD *)(a1 + 16);
    if ( v52 )
      v30 = v52;
    ++*(_QWORD *)a1;
    v54 = v53 + 1;
    if ( 4 * v54 >= 3 * v26 )
      goto LABEL_103;
    v55 = v70 - 24;
    if ( v26 - *(_DWORD *)(a1 + 20) - v54 <= v26 >> 3 )
      goto LABEL_104;
LABEL_99:
    *(_DWORD *)(a1 + 16) = v54;
    if ( *v30 != -8 )
      --*(_DWORD *)(a1 + 20);
    *v30 = v55;
    v30[1] = 0;
LABEL_39:
    v30[1] = v68;
    v32 = *(__int64 **)(a1 + 72);
    if ( *(__int64 **)(a1 + 80) == v32 )
    {
      v47 = &v32[*(unsigned int *)(a1 + 92)];
      v48 = *(_DWORD *)(a1 + 92);
      if ( v32 == v47 )
      {
LABEL_91:
        if ( v48 >= *(_DWORD *)(a1 + 88) )
          goto LABEL_40;
        *(_DWORD *)(a1 + 92) = v48 + 1;
        *v47 = v27;
        ++*(_QWORD *)(a1 + 64);
      }
      else
      {
        v49 = 0;
        while ( v27 != *v32 )
        {
          if ( *v32 == -2 )
            v49 = v32;
          if ( v47 == ++v32 )
          {
            if ( !v49 )
              goto LABEL_91;
            *v49 = v27;
            --*(_DWORD *)(a1 + 96);
            ++*(_QWORD *)(a1 + 64);
            break;
          }
        }
      }
    }
    else
    {
LABEL_40:
      sub_16CCBA0(v56, v27);
    }
    v72 = v73;
LABEL_42:
    v70 = *(_QWORD *)(v70 + 8);
  }
  while ( v62 != v70 );
  return v72;
}
