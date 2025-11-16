// Function: sub_2722E50
// Address: 0x2722e50
//
unsigned __int64 __fastcall sub_2722E50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r8
  __int64 v8; // rsi
  _QWORD *v9; // rax
  _QWORD *v10; // rcx
  _QWORD *v11; // rbx
  _QWORD *v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // r12
  __int64 v20; // rbx
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned __int8 v23; // dl
  __int64 v24; // rsi
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // r9
  unsigned int v29; // r14d
  __int64 v30; // rcx
  __int64 v31; // rdx
  unsigned __int8 *v32; // rdi
  _QWORD *v33; // rax
  _QWORD *v34; // rdx
  _QWORD *v35; // rax
  __int64 *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  int v40; // r11d
  __int64 v41; // rax
  int v42; // ecx
  int v43; // edx
  __int64 v44; // rax
  __int64 *v45; // rdi
  __int64 *v46; // r8
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  _QWORD **v51; // rbx
  _QWORD **v52; // r12
  _QWORD *v53; // rdi
  bool v54; // al
  __int64 v56; // rax
  int v57; // eax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  unsigned __int8 v61; // dl
  __int64 v62; // rsi
  _QWORD *v63; // rax
  _QWORD *v64; // rdx
  int v65; // eax
  __int64 v66; // r10
  unsigned int v67; // ecx
  int v68; // edi
  __int64 v69; // rsi
  int v70; // r11d
  int v71; // r11d
  int v72; // esi
  __int64 v73; // rcx
  __int64 v74; // rdi
  _QWORD *v75; // [rsp+0h] [rbp-90h]
  unsigned __int8 v76; // [rsp+Eh] [rbp-82h]
  char v77; // [rsp+Fh] [rbp-81h]
  _QWORD *v78; // [rsp+10h] [rbp-80h]
  _QWORD *v79; // [rsp+18h] [rbp-78h]
  __int64 *v80; // [rsp+20h] [rbp-70h]
  __int64 v81; // [rsp+28h] [rbp-68h]
  unsigned __int64 v82; // [rsp+30h] [rbp-60h]
  _QWORD *v83; // [rsp+38h] [rbp-58h]
  unsigned __int64 v84; // [rsp+40h] [rbp-50h]
  unsigned __int8 *v85; // [rsp+48h] [rbp-48h]
  __int64 v86[7]; // [rsp+58h] [rbp-38h] BYREF

  v76 = sub_2721A90(a1, a2, a3, a4, a5, a6);
  v8 = *(_QWORD *)a1 + 72LL;
  v9 = *(_QWORD **)(*(_QWORD *)a1 + 80LL);
  v83 = (_QWORD *)v8;
  v78 = v9;
  if ( (_QWORD *)v8 == v9 )
  {
    v75 = 0;
  }
  else
  {
    if ( !v9 )
      BUG();
    while ( 1 )
    {
      v10 = (_QWORD *)v9[4];
      if ( v10 != v9 + 3 )
        break;
      v9 = (_QWORD *)v9[1];
      if ( (_QWORD *)v8 == v9 )
        break;
      if ( !v9 )
        BUG();
    }
    v75 = v10;
    v78 = v9;
  }
  v84 = 0;
  v11 = (_QWORD *)(*(_QWORD *)a1 + 72LL);
  v77 = 0;
  v81 = a1 + 1144;
  while ( 1 )
  {
    v12 = (_QWORD *)v84;
    v13 = v11;
    if ( v11 == v78 && (v11 == v83 || (_QWORD *)v84 == v75) )
      break;
    while ( 1 )
    {
      if ( v83 != v13 )
      {
        if ( !v13 )
          BUG();
        if ( v12 != (_QWORD *)v13[4] )
          break;
      }
      v13 = (_QWORD *)(*v13 & 0xFFFFFFFFFFFFFFF8LL);
      v14 = v13 - 3;
      if ( !v13 )
        v14 = 0;
      v12 = v14 + 6;
    }
    v15 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
    v82 = v15;
    if ( !v15 )
      BUG();
    v16 = *(_QWORD *)(v15 + 40);
    v85 = (unsigned __int8 *)(v15 - 24);
    if ( v16 )
    {
      v18 = sub_B14240(v16);
      if ( v18 != v17 )
      {
        v79 = v11;
        v19 = v17;
        while ( 2 )
        {
          v20 = v18;
          v18 = *(_QWORD *)(v18 + 8);
          if ( !*(_BYTE *)(v20 + 32) && *(_BYTE *)(v20 + 64) == 2 )
          {
            v37 = sub_B13870(v20);
            v38 = sub_AE9410(v37);
            if ( v39 != v38 )
              goto LABEL_33;
          }
          v21 = *(_QWORD *)(v20 + 24);
          v86[0] = v21;
          if ( v21 )
            sub_B96E90((__int64)v86, v21, 1);
          v22 = sub_B10CD0((__int64)v86);
          v23 = *(_BYTE *)(v22 - 16);
          if ( (v23 & 2) != 0 )
          {
            v24 = **(_QWORD **)(v22 - 32);
            if ( *(_BYTE *)(a1 + 1172) )
              goto LABEL_27;
LABEL_47:
            v36 = sub_C8CA60(v81, v24);
            if ( v86[0] )
            {
              v80 = v36;
              sub_B91220((__int64)v86, v86[0]);
              v36 = v80;
            }
            if ( v36 )
              goto LABEL_33;
          }
          else
          {
            v24 = *(_QWORD *)(v22 - 16 - 8LL * ((v23 >> 2) & 0xF));
            if ( !*(_BYTE *)(a1 + 1172) )
              goto LABEL_47;
LABEL_27:
            v25 = *(_QWORD **)(a1 + 1152);
            v26 = &v25[*(unsigned int *)(a1 + 1164)];
            if ( v25 != v26 )
            {
              while ( v24 != *v25 )
              {
                if ( v26 == ++v25 )
                  goto LABEL_51;
              }
              if ( v86[0] )
                sub_B91220((__int64)v86, v86[0]);
              goto LABEL_33;
            }
LABEL_51:
            if ( v86[0] )
              sub_B91220((__int64)v86, v86[0]);
          }
          sub_B44590((__int64)v85, (_QWORD *)v20);
LABEL_33:
          if ( v18 == v19 )
          {
            v11 = v79;
            break;
          }
          continue;
        }
      }
    }
    v27 = *(_DWORD *)(a1 + 96);
    if ( !v27 )
    {
      ++*(_QWORD *)(a1 + 72);
      goto LABEL_103;
    }
    v28 = *(_QWORD *)(a1 + 80);
    v29 = ((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4);
    LODWORD(v30) = (v27 - 1) & v29;
    v31 = v28 + 24LL * (unsigned int)v30;
    v32 = *(unsigned __int8 **)v31;
    if ( v85 != *(unsigned __int8 **)v31 )
    {
      v40 = 1;
      v41 = 0;
      while ( v32 != (unsigned __int8 *)-4096LL )
      {
        if ( v32 == (unsigned __int8 *)-8192LL && !v41 )
          v41 = v31;
        v7 = (unsigned int)(v40 + 1);
        v30 = (v27 - 1) & ((_DWORD)v30 + v40);
        v31 = v28 + 24 * v30;
        v32 = *(unsigned __int8 **)v31;
        if ( v85 == *(unsigned __int8 **)v31 )
          goto LABEL_37;
        ++v40;
      }
      v42 = *(_DWORD *)(a1 + 88);
      if ( !v41 )
        v41 = v31;
      ++*(_QWORD *)(a1 + 72);
      v43 = v42 + 1;
      if ( 4 * (v42 + 1) >= 3 * v27 )
      {
LABEL_103:
        sub_271E3C0(a1 + 72, 2 * v27);
        v65 = *(_DWORD *)(a1 + 96);
        if ( !v65 )
          goto LABEL_134;
        v7 = (unsigned int)(v65 - 1);
        v66 = *(_QWORD *)(a1 + 80);
        v43 = *(_DWORD *)(a1 + 88) + 1;
        v67 = v7 & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
        v41 = v66 + 24LL * v67;
        v28 = *(_QWORD *)v41;
        if ( v85 != *(unsigned __int8 **)v41 )
        {
          v68 = 1;
          v69 = 0;
          while ( v28 != -4096 )
          {
            if ( v28 == -8192 && !v69 )
              v69 = v41;
            v67 = v7 & (v68 + v67);
            v41 = v66 + 24LL * v67;
            v28 = *(_QWORD *)v41;
            if ( v85 == *(unsigned __int8 **)v41 )
              goto LABEL_62;
            ++v68;
          }
          if ( v69 )
            v41 = v69;
        }
      }
      else if ( v27 - *(_DWORD *)(a1 + 92) - v43 <= v27 >> 3 )
      {
        sub_271E3C0(a1 + 72, v27);
        v70 = *(_DWORD *)(a1 + 96);
        if ( !v70 )
        {
LABEL_134:
          ++*(_DWORD *)(a1 + 88);
          BUG();
        }
        v71 = v70 - 1;
        v28 = *(_QWORD *)(a1 + 80);
        v7 = v71 & v29;
        v72 = 1;
        v43 = *(_DWORD *)(a1 + 88) + 1;
        v73 = 0;
        v41 = v28 + 24 * v7;
        v74 = *(_QWORD *)v41;
        if ( v85 != *(unsigned __int8 **)v41 )
        {
          while ( v74 != -4096 )
          {
            if ( v74 == -8192 && !v73 )
              v73 = v41;
            v7 = v71 & (unsigned int)(v72 + v7);
            v41 = v28 + 24LL * (unsigned int)v7;
            v74 = *(_QWORD *)v41;
            if ( v85 == *(unsigned __int8 **)v41 )
              goto LABEL_62;
            ++v72;
          }
          if ( v73 )
            v41 = v73;
        }
      }
LABEL_62:
      *(_DWORD *)(a1 + 88) = v43;
      if ( *(_QWORD *)v41 != -4096 )
        --*(_DWORD *)(a1 + 92);
      *(_BYTE *)(v41 + 8) = 0;
      *(_QWORD *)(v41 + 16) = 0;
      *(_QWORD *)v41 = v85;
      goto LABEL_65;
    }
LABEL_37:
    if ( *(_BYTE *)(v31 + 8) )
      goto LABEL_38;
LABEL_65:
    if ( *(_BYTE *)(v82 - 24) != 85
      || (v56 = *(_QWORD *)(v82 - 56)) == 0
      || *(_BYTE *)v56
      || *(_QWORD *)(v56 + 24) != *(_QWORD *)(v82 + 56)
      || (*(_BYTE *)(v56 + 33) & 0x20) == 0
      || (v57 = *(_DWORD *)(v56 + 36), (unsigned int)(v57 - 68) > 3) )
    {
      v77 = 1;
      goto LABEL_67;
    }
    if ( v57 == 68 )
    {
      v58 = sub_AE9410(*(_QWORD *)(*(_QWORD *)(v82 + 32 * (3LL - (*(_DWORD *)(v82 - 20) & 0x7FFFFFF)) - 24) + 24LL));
      if ( v58 != v59 )
        goto LABEL_38;
    }
    v60 = sub_B10CD0(v82 + 24);
    v61 = *(_BYTE *)(v60 - 16);
    if ( (v61 & 2) != 0 )
    {
      v62 = **(_QWORD **)(v60 - 32);
      if ( !*(_BYTE *)(a1 + 1172) )
        goto LABEL_117;
    }
    else
    {
      v62 = *(_QWORD *)(v60 - 16 - 8LL * ((v61 >> 2) & 0xF));
      if ( !*(_BYTE *)(a1 + 1172) )
      {
LABEL_117:
        if ( sub_C8CA60(v81, v62) )
          goto LABEL_38;
        goto LABEL_67;
      }
    }
    v63 = *(_QWORD **)(a1 + 1152);
    v64 = &v63[*(unsigned int *)(a1 + 1164)];
    if ( v63 != v64 )
    {
      while ( v62 != *v63 )
      {
        if ( v64 == ++v63 )
          goto LABEL_67;
      }
LABEL_38:
      v33 = (_QWORD *)v84;
      v34 = v83;
      goto LABEL_42;
    }
LABEL_67:
    v44 = *(unsigned int *)(a1 + 112);
    if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 116) )
    {
      sub_C8D5F0(a1 + 104, (const void *)(a1 + 120), v44 + 1, 8u, v7, v28);
      v44 = *(unsigned int *)(a1 + 112);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * v44) = v85;
    ++*(_DWORD *)(a1 + 112);
    sub_F54ED0(v85);
    v33 = (_QWORD *)v84;
    v34 = v83;
LABEL_42:
    while ( 1 )
    {
      if ( v11 != v34 )
      {
        if ( !v11 )
          BUG();
        if ( (_QWORD *)v11[4] != v33 )
          break;
      }
      v11 = (_QWORD *)(*v11 & 0xFFFFFFFFFFFFFFF8LL);
      v35 = v11 - 3;
      if ( !v11 )
        v35 = 0;
      v33 = v35 + 6;
    }
    v83 = v34;
    v84 = *v33 & 0xFFFFFFFFFFFFFFF8LL;
  }
  v45 = *(__int64 **)(a1 + 104);
  v46 = &v45[*(unsigned int *)(a1 + 112)];
  if ( v46 != v45 )
  {
    do
    {
      v47 = *v45;
      v48 = 32LL * (*(_DWORD *)(*v45 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(*v45 + 7) & 0x40) != 0 )
      {
        v49 = *(_QWORD *)(v47 - 8);
        v47 = v49 + v48;
      }
      else
      {
        v49 = v47 - v48;
      }
      for ( ; v47 != v49; v49 += 32 )
      {
        if ( *(_QWORD *)v49 )
        {
          v50 = *(_QWORD *)(v49 + 8);
          **(_QWORD **)(v49 + 16) = v50;
          if ( v50 )
            *(_QWORD *)(v50 + 16) = *(_QWORD *)(v49 + 16);
        }
        *(_QWORD *)v49 = 0;
      }
      ++v45;
    }
    while ( v46 != v45 );
    v51 = *(_QWORD ***)(a1 + 104);
    v52 = &v51[*(unsigned int *)(a1 + 112)];
    while ( v52 != v51 )
    {
      v53 = *v51++;
      sub_B43D60(v53);
    }
  }
  v54 = 1;
  if ( !v76 )
    v54 = *(_DWORD *)(a1 + 112) != 0;
  LOBYTE(v86[0]) = v54;
  BYTE1(v86[0]) = v77;
  return LOWORD(v86[0]) | ((unsigned __int64)v76 << 16);
}
