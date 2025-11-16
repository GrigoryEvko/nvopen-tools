// Function: sub_143B970
// Address: 0x143b970
//
__int64 __fastcall sub_143B970(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // rax
  _QWORD *v7; // r15
  char *v11; // rsi
  unsigned __int8 v12; // r8
  __int64 *v13; // r9
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  _QWORD *v19; // rax
  int v20; // r11d
  __int64 v21; // r11
  __int64 v22; // rax
  char v23; // di
  unsigned int v24; // esi
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r9
  unsigned int v29; // eax
  __int64 v30; // r15
  __int64 *v31; // rax
  __int64 v32; // rdx
  unsigned int v33; // eax
  __int64 v34; // r12
  __int64 v35; // rcx
  __int64 v36; // r8
  char v37; // al
  __int64 v38; // r15
  __int64 v39; // r8
  __int64 v40; // r11
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // rax
  char v45; // r15
  char v46; // r11
  _QWORD *v47; // rax
  __int64 v48; // rax
  char v49; // r11
  __int64 v50; // rsi
  __int64 v51; // r15
  char *v52; // rax
  __int64 v53; // rdx
  char *v54; // rcx
  __int64 v55; // rdx
  char *v56; // rdx
  __int64 **v57; // rcx
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 **v60; // rdi
  __int64 v61; // rcx
  __int64 v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // rdi
  __int64 v65; // rbx
  unsigned __int64 v66; // r13
  __int64 *v67; // rdi
  __int64 v68; // rax
  __int64 v69; // r14
  __int64 v70; // rax
  __int64 v71; // r8
  int v72; // eax
  __int64 v73; // rdx
  __int64 v74; // rsi
  __int64 v75; // rcx
  __int64 **v76; // rdx
  __int64 **v77; // rcx
  __int64 **v78; // rax
  _QWORD *v79; // rax
  __int64 i; // r13
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // rax
  __int64 v85; // rsi
  signed __int64 v86; // rdx
  __int64 *v87; // [rsp+0h] [rbp-F0h]
  __int64 v88; // [rsp+8h] [rbp-E8h]
  char v89; // [rsp+10h] [rbp-E0h]
  __int64 v90; // [rsp+10h] [rbp-E0h]
  __int64 v91; // [rsp+18h] [rbp-D8h]
  __int64 v92; // [rsp+18h] [rbp-D8h]
  __int64 v93; // [rsp+20h] [rbp-D0h]
  char v94; // [rsp+20h] [rbp-D0h]
  __int64 v95; // [rsp+20h] [rbp-D0h]
  __int64 v97; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v98[6]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 **v99; // [rsp+70h] [rbp-80h] BYREF
  __int64 v100; // [rsp+78h] [rbp-78h]
  _QWORD v101[14]; // [rsp+80h] [rbp-70h] BYREF

  v5 = a2;
  if ( *(_BYTE *)(a2 + 16) <= 0x17u )
    return v5;
  v6 = *(unsigned int *)(a1 + 40);
  v7 = *(_QWORD **)(a1 + 32);
  v97 = a2;
  v11 = (char *)&v7[v6];
  if ( v11 != (char *)sub_143B670(v7, (__int64)v11, &v97) )
  {
    if ( a3 != *(_QWORD *)(v5 + 40) )
      return v5;
    v93 = a1 + 32;
    v19 = sub_143B670(v7, (__int64)v11, v13);
    if ( v11 != (char *)(v19 + 1) )
    {
      memmove(v19, v19 + 1, v11 - (char *)(v19 + 1));
      v5 = v97;
      v20 = *(_DWORD *)(a1 + 40);
    }
    v21 = (unsigned int)(v20 - 1);
    *(_DWORD *)(a1 + 40) = v21;
    if ( *(_BYTE *)(v5 + 16) == 77 )
    {
      v22 = 0x17FFFFFFE8LL;
      v23 = *(_BYTE *)(v5 + 23) & 0x40;
      v24 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
      if ( v24 )
      {
        v25 = 24LL * *(unsigned int *)(v5 + 56) + 8;
        v26 = 0;
        while ( 1 )
        {
          v27 = v5 - 24LL * v24;
          if ( v23 )
            v27 = *(_QWORD *)(v5 - 8);
          if ( a4 == *(_QWORD *)(v27 + v25) )
            break;
          ++v26;
          v25 += 8;
          if ( v24 == (_DWORD)v26 )
          {
            v22 = 0x17FFFFFFE8LL;
            goto LABEL_26;
          }
        }
        v22 = 24 * v26;
      }
LABEL_26:
      if ( v23 )
        v28 = *(_QWORD *)(v5 - 8);
      else
        v28 = v5 - 24LL * v24;
      v16 = *(_QWORD *)(v28 + v22);
      if ( !v16 )
        BUG();
      if ( *(_BYTE *)(v16 + 16) > 0x17u )
      {
        if ( (unsigned int)v21 >= *(_DWORD *)(a1 + 44) )
        {
          sub_16CD150(v93, a1 + 48, 0, 8);
          v21 = *(unsigned int *)(a1 + 40);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v21) = v16;
        ++*(_DWORD *)(a1 + 40);
      }
      return v16;
    }
    if ( !(unsigned __int8)sub_143B820(v5) )
      return 0;
    if ( (*(_DWORD *)(v5 + 20) & 0xFFFFFFF) != 0 )
    {
      v38 = 0;
      v39 = a4;
      v40 = 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
      do
      {
        if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
          v41 = *(_QWORD *)(v5 - 8);
        else
          v41 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
        v42 = *(_QWORD *)(v41 + v38);
        if ( *(_BYTE *)(v42 + 16) > 0x17u )
        {
          v43 = *(unsigned int *)(a1 + 40);
          if ( (unsigned int)v43 >= *(_DWORD *)(a1 + 44) )
          {
            v88 = v40;
            v90 = v39;
            sub_16CD150(v93, a1 + 48, 0, 8);
            v43 = *(unsigned int *)(a1 + 40);
            v40 = v88;
            v39 = v90;
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v43) = v42;
          ++*(_DWORD *)(a1 + 40);
        }
        v38 += 24;
      }
      while ( v40 != v38 );
      a4 = v39;
    }
    v12 = *(_BYTE *)(v5 + 16);
  }
  if ( (unsigned int)v12 - 60 <= 0xC )
  {
    if ( (unsigned __int8)sub_14AF470(v5, 0, 0, 0) )
    {
      v14 = sub_143B970(a1, *(_QWORD *)(v5 - 24), a3, a4, a5);
      if ( v14 )
      {
        v15 = *(_QWORD *)(v5 - 24);
        if ( v15 == v14 )
        {
          v16 = v5;
          if ( v15 )
            return v16;
        }
        if ( *(_BYTE *)(v14 + 16) <= 0x10u )
        {
          v16 = sub_15A46C0((unsigned int)*(unsigned __int8 *)(v5 + 16) - 24, v14, *(_QWORD *)v5, 0);
          if ( *(_BYTE *)(v16 + 16) <= 0x17u )
            return v16;
          v17 = *(unsigned int *)(a1 + 40);
          if ( (unsigned int)v17 < *(_DWORD *)(a1 + 44) )
          {
LABEL_12:
            *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v17) = v16;
            ++*(_DWORD *)(a1 + 40);
            return v16;
          }
LABEL_11:
          sub_16CD150(a1 + 32, a1 + 48, 0, 8);
          v17 = *(unsigned int *)(a1 + 40);
          goto LABEL_12;
        }
        v34 = *(_QWORD *)(v14 + 8);
        if ( v34 )
        {
          while ( 1 )
          {
            v16 = sub_1648700(v34);
            v37 = *(_BYTE *)(v16 + 16);
            if ( (unsigned __int8)(v37 - 60) <= 0xCu
              && v37 == *(_BYTE *)(v5 + 16)
              && *(_QWORD *)v16 == *(_QWORD *)v5
              && (!a5 || (unsigned __int8)sub_15CC8F0(a5, *(_QWORD *)(v16 + 40), a4, v35, v36)) )
            {
              break;
            }
            v34 = *(_QWORD *)(v34 + 8);
            if ( !v34 )
              return 0;
          }
          return v16;
        }
      }
    }
    return 0;
  }
  if ( v12 != 56 )
  {
    if ( v12 != 35 )
      return 0;
    v44 = (*(_BYTE *)(v5 + 23) & 0x40) != 0 ? *(_QWORD *)(v5 - 8) : v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
    v92 = *(_QWORD *)(v44 + 24);
    if ( *(_BYTE *)(v92 + 16) != 13 )
      return 0;
    v45 = sub_15F2380(v5);
    v46 = sub_15F2370(v5);
    v47 = (*(_BYTE *)(v5 + 23) & 0x40) != 0
        ? *(_QWORD **)(v5 - 8)
        : (_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
    v89 = v46;
    v48 = sub_143B970(a1, *v47, a3, a4, a5);
    v95 = v48;
    if ( !v48 )
      return 0;
    v49 = v89;
    if ( *(_BYTE *)(v48 + 16) != 35 || (v50 = *(_QWORD *)(v48 - 24), *(_BYTE *)(v50 + 16) != 13) )
    {
LABEL_81:
      v57 = *(__int64 ***)(a1 + 8);
      v58 = *(_QWORD *)(a1 + 16);
      v101[2] = 0;
      v59 = *(_QWORD *)(a1 + 24);
      v99 = v57;
      v100 = v58;
      v101[0] = a5;
      v101[1] = v59;
      v16 = (__int64)sub_13DEB20(v95, v92, v45, v49, &v99);
      if ( v16 )
      {
        sub_143B730(v95, a1 + 32);
        if ( *(_BYTE *)(v16 + 16) <= 0x17u )
          return v16;
        v17 = *(unsigned int *)(a1 + 40);
        if ( (unsigned int)v17 < *(_DWORD *)(a1 + 44) )
          goto LABEL_12;
        goto LABEL_11;
      }
      if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
        v79 = *(_QWORD **)(v5 - 8);
      else
        v79 = (_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
      if ( v95 != *v79 || v92 != v79[3] )
      {
        for ( i = *(_QWORD *)(v95 + 8); i; i = *(_QWORD *)(i + 8) )
        {
          v81 = sub_1648700(i);
          v16 = v81;
          if ( *(_BYTE *)(v81 + 16) == 35 )
          {
            v84 = *(_QWORD *)(v81 - 48);
            if ( v95 == v84 )
            {
              if ( v84 )
              {
                if ( v92 == *(_QWORD *)(v16 - 24) )
                {
                  v85 = *(_QWORD *)(v16 + 40);
                  if ( *(_QWORD *)(a3 + 56) == *(_QWORD *)(v85 + 56)
                    && (!a5 || (unsigned __int8)sub_15CC8F0(a5, v85, a4, v82, v83)) )
                  {
                    return v16;
                  }
                }
              }
            }
          }
        }
        return 0;
      }
      return v5;
    }
    v51 = *(_QWORD *)(v48 - 48);
    v92 = sub_15A2B30(v92, v50, 0, 0);
    v52 = *(char **)(a1 + 32);
    v53 = 8LL * *(unsigned int *)(a1 + 40);
    v54 = &v52[v53];
    v55 = v53 >> 5;
    if ( v55 )
    {
      v56 = &v52[32 * v55];
      while ( 1 )
      {
        if ( v95 == *(_QWORD *)v52 )
          goto LABEL_79;
        if ( v95 == *((_QWORD *)v52 + 1) )
        {
          v52 += 8;
          goto LABEL_79;
        }
        if ( v95 == *((_QWORD *)v52 + 2) )
        {
          v52 += 16;
          goto LABEL_79;
        }
        if ( v95 == *((_QWORD *)v52 + 3) )
          break;
        v52 += 32;
        if ( v56 == v52 )
          goto LABEL_144;
      }
      v52 += 24;
LABEL_79:
      if ( v54 != v52 )
      {
        sub_143B730(v95, a1 + 32);
        if ( *(_BYTE *)(v51 + 16) > 0x17u )
        {
          if ( *(_DWORD *)(a1 + 40) >= *(_DWORD *)(a1 + 44) )
            sub_16CD150(a1 + 32, a1 + 48, 0, 8);
          v95 = v51;
          v49 = 0;
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40)) = v51;
          v45 = 0;
          ++*(_DWORD *)(a1 + 40);
          goto LABEL_81;
        }
      }
LABEL_80:
      v95 = v51;
      v49 = 0;
      v45 = 0;
      goto LABEL_81;
    }
LABEL_144:
    v86 = v54 - v52;
    if ( v54 - v52 != 16 )
    {
      if ( v86 != 24 )
      {
        if ( v86 != 8 )
          goto LABEL_80;
LABEL_147:
        if ( v95 == *(_QWORD *)v52 )
          goto LABEL_79;
        goto LABEL_80;
      }
      if ( v95 == *(_QWORD *)v52 )
        goto LABEL_79;
      v52 += 8;
    }
    if ( v95 == *(_QWORD *)v52 )
      goto LABEL_79;
    v52 += 8;
    goto LABEL_147;
  }
  v99 = (__int64 **)v101;
  v100 = 0x800000000LL;
  v29 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
  if ( !v29 )
    return v5;
  v94 = 0;
  v30 = 0;
  v91 = v29 - 1;
  while ( 1 )
  {
    v31 = (__int64 *)sub_143B970(a1, *(_QWORD *)(v5 + 24 * (v30 - v29)), a3, a4, a5);
    if ( !v31 )
    {
LABEL_87:
      v60 = v99;
LABEL_88:
      v16 = 0;
      goto LABEL_89;
    }
    v94 |= *(_QWORD *)(v5 + 24 * (v30 - (*(_DWORD *)(v5 + 20) & 0xFFFFFFF))) != (_QWORD)v31;
    v32 = (unsigned int)v100;
    if ( (unsigned int)v100 >= HIDWORD(v100) )
    {
      v87 = v31;
      sub_16CD150(&v99, v101, 0, 8);
      v32 = (unsigned int)v100;
      v31 = v87;
    }
    v99[v32] = v31;
    v33 = v100 + 1;
    LODWORD(v100) = v100 + 1;
    if ( v91 == v30 )
      break;
    ++v30;
    v29 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
  }
  if ( !v94 )
  {
    v16 = v5;
LABEL_103:
    v60 = v99;
    goto LABEL_89;
  }
  v61 = *(_QWORD *)(a1 + 16);
  v62 = *(_QWORD *)(a1 + 8);
  v98[4] = 0;
  v63 = *(_QWORD *)(a1 + 24);
  v64 = *(_QWORD *)(v5 + 56);
  v98[1] = v61;
  v98[0] = v62;
  v98[2] = a5;
  v98[3] = v63;
  v16 = sub_13E3340(v64, v99, v33, v98);
  if ( !v16 )
  {
    v60 = v99;
    v69 = (*v99)[1];
    if ( !v69 )
      goto LABEL_88;
    while ( 1 )
    {
      v70 = sub_1648700(v69);
      v16 = v70;
      if ( *(_BYTE *)(v70 + 16) != 56 )
        goto LABEL_107;
      if ( *(_QWORD *)v5 != *(_QWORD *)v70 )
        goto LABEL_107;
      v72 = *(_DWORD *)(v70 + 20);
      v73 = (unsigned int)v100;
      if ( (*(_DWORD *)(v16 + 20) & 0xFFFFFFF) != (unsigned __int64)(unsigned int)v100 )
        goto LABEL_107;
      v74 = *(_QWORD *)(v16 + 40);
      v75 = *(_QWORD *)(v74 + 56);
      if ( *(_QWORD *)(a3 + 56) != v75 )
        goto LABEL_107;
      if ( a5 )
      {
        if ( !(unsigned __int8)sub_15CC8F0(a5, v74, a4, v75, v71) )
          goto LABEL_107;
        v72 = *(_DWORD *)(v16 + 20);
        v73 = (unsigned int)v100;
      }
      v60 = v99;
      v76 = &v99[v73];
      v77 = (__int64 **)(v16 - 24LL * (v72 & 0xFFFFFFF));
      v78 = v99;
      if ( v76 == v99 )
        goto LABEL_89;
      while ( *v78 == *v77 )
      {
        ++v78;
        v77 += 3;
        if ( v76 == v78 )
          goto LABEL_89;
      }
LABEL_107:
      v69 = *(_QWORD *)(v69 + 8);
      if ( !v69 )
        goto LABEL_87;
    }
  }
  if ( (_DWORD)v100 )
  {
    v65 = 8LL * (unsigned int)v100;
    v66 = 0;
    do
    {
      v67 = v99[v66 / 8];
      v66 += 8LL;
      sub_143B730((__int64)v67, a1 + 32);
    }
    while ( v65 != v66 );
  }
  if ( *(_BYTE *)(v16 + 16) <= 0x17u )
    goto LABEL_103;
  v68 = *(unsigned int *)(a1 + 40);
  if ( (unsigned int)v68 >= *(_DWORD *)(a1 + 44) )
  {
    sub_16CD150(a1 + 32, a1 + 48, 0, 8);
    v68 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v68) = v16;
  v60 = v99;
  ++*(_DWORD *)(a1 + 40);
LABEL_89:
  if ( v60 != v101 )
    _libc_free((unsigned __int64)v60);
  return v16;
}
