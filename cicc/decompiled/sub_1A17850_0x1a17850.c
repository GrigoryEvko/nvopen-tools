// Function: sub_1A17850
// Address: 0x1a17850
//
__int64 __fastcall sub_1A17850(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rax
  __int64 v5; // r14
  __int64 v6; // r12
  unsigned __int8 v7; // al
  unsigned __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // r15
  char v11; // al
  __int64 *v12; // rax
  __int64 v13; // rbx
  __int64 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // r13
  __int64 **v18; // r11
  __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  __int64 *v21; // rsi
  __int64 *v22; // rdx
  int v23; // ebx
  unsigned int i; // r13d
  __int64 *v25; // rax
  int v26; // r8d
  int v27; // r9d
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // r15
  __int64 v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // eax
  unsigned __int64 v38; // rbx
  unsigned int v39; // r13d
  unsigned __int64 v40; // r13
  unsigned int v41; // eax
  unsigned __int64 v42; // r12
  char v43; // al
  __int64 v44; // rdx
  char v45; // al
  __int64 v46; // rax
  __int64 *v47; // rax
  _QWORD *v48; // rax
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rax
  __int64 v51; // rcx
  int v52; // eax
  int v53; // esi
  __int64 v54; // rdi
  unsigned int v55; // eax
  __int64 v56; // rdx
  int v57; // r9d
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 *v63; // rdx
  _QWORD *v64; // rax
  unsigned __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 *v70; // rax
  __int64 v71; // rsi
  unsigned __int64 v72; // rdi
  __int64 v73; // rsi
  __int64 v74; // rax
  __int64 *v75; // rax
  __int64 v76; // rax
  __int64 v77; // rcx
  unsigned __int64 v78; // rsi
  __int64 v79; // rcx
  __int64 v80; // rsi
  __int64 v81; // rdx
  unsigned __int64 v82; // rdi
  __int64 v83; // rsi
  unsigned __int64 v84; // r12
  __int64 **v85; // [rsp+8h] [rbp-68h]
  __int64 **v86; // [rsp+8h] [rbp-68h]
  __int64 **v87; // [rsp+8h] [rbp-68h]
  const void *v88; // [rsp+10h] [rbp-60h]
  __int64 v89; // [rsp+18h] [rbp-58h]
  __int64 v90; // [rsp+20h] [rbp-50h]
  __int64 v91; // [rsp+28h] [rbp-48h]
  __int64 v92; // [rsp+30h] [rbp-40h]

  v89 = a2 + 72;
  v92 = *(_QWORD *)(a2 + 80);
  if ( v92 == a2 + 72 )
    return 0;
  v2 = a1;
  v90 = a1 + 16;
  v88 = (const void *)(a1 + 832);
  while ( 1 )
  {
    v3 = 0;
    if ( v92 )
      v3 = v92 - 24;
    v91 = v3;
    if ( !sub_183E920(v90, v3) )
      goto LABEL_6;
    v5 = *(_QWORD *)(v91 + 48);
    if ( v5 != v91 + 40 )
      break;
LABEL_64:
    v42 = sub_157EBA0(v91);
    v43 = *(_BYTE *)(v42 + 16);
    if ( v43 == 26 )
    {
      if ( (*(_DWORD *)(v42 + 20) & 0xFFFFFFF) == 3 && (*(_BYTE *)sub_1A10F60(v2, *(_QWORD *)(v42 - 72)) & 6) == 0 )
      {
        if ( *(_BYTE *)(*(_QWORD *)(v42 - 72) + 16LL) != 9 )
        {
          v44 = sub_15F4DF0(v42, 1u);
          goto LABEL_69;
        }
        v75 = (__int64 *)sub_16498A0(v42);
        v76 = sub_159C540(v75);
        if ( *(_QWORD *)(v42 - 72) )
        {
          v77 = *(_QWORD *)(v42 - 64);
          v78 = *(_QWORD *)(v42 - 56) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v78 = v77;
          if ( v77 )
            *(_QWORD *)(v77 + 16) = v78 | *(_QWORD *)(v77 + 16) & 3LL;
        }
        *(_QWORD *)(v42 - 72) = v76;
        if ( v76 )
        {
          v79 = *(_QWORD *)(v76 + 8);
          *(_QWORD *)(v42 - 64) = v79;
          if ( v79 )
            *(_QWORD *)(v79 + 16) = (v42 - 64) | *(_QWORD *)(v79 + 16) & 3LL;
          *(_QWORD *)(v42 - 56) = (v76 + 8) | *(_QWORD *)(v42 - 56) & 3LL;
          *(_QWORD *)(v76 + 8) = v42 - 72;
        }
        v74 = sub_15F4DF0(v42, 1u);
        goto LABEL_151;
      }
    }
    else if ( v43 == 28 )
    {
      v46 = *(_DWORD *)(v42 + 20) & 0xFFFFFFF;
      if ( (_DWORD)v46 != 1 )
      {
        v47 = (*(_BYTE *)(v42 + 23) & 0x40) != 0 ? *(__int64 **)(v42 - 8) : (__int64 *)(v42 - 24 * v46);
        if ( (*(_BYTE *)sub_1A10F60(v2, *v47) & 6) == 0 )
        {
          if ( (*(_BYTE *)(v42 + 23) & 0x40) != 0 )
            v48 = *(_QWORD **)(v42 - 8);
          else
            v48 = (_QWORD *)(v42 - 24LL * (*(_DWORD *)(v42 + 20) & 0xFFFFFFF));
          if ( *(_BYTE *)(*v48 + 16LL) != 9 )
          {
            v44 = v48[3];
LABEL_69:
            if ( (unsigned __int8)sub_1A153A0(v2, v91, v44) )
              return 1;
            goto LABEL_6;
          }
          v68 = sub_1A13110(v42, 1u);
          v69 = sub_159BF40(v68);
          if ( (*(_BYTE *)(v42 + 23) & 0x40) != 0 )
            v70 = *(__int64 **)(v42 - 8);
          else
            v70 = (__int64 *)(v42 - 24LL * (*(_DWORD *)(v42 + 20) & 0xFFFFFFF));
          if ( *v70 )
          {
            v71 = v70[1];
            v72 = v70[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v72 = v71;
            if ( v71 )
              *(_QWORD *)(v71 + 16) = v72 | *(_QWORD *)(v71 + 16) & 3LL;
          }
          *v70 = v69;
          if ( v69 )
          {
            v73 = *(_QWORD *)(v69 + 8);
            v70[1] = v73;
            if ( v73 )
              *(_QWORD *)(v73 + 16) = (unsigned __int64)(v70 + 1) | *(_QWORD *)(v73 + 16) & 3LL;
            v70[2] = (v69 + 8) | v70[2] & 3;
            *(_QWORD *)(v69 + 8) = v70;
          }
          v74 = sub_1A13110(v42, 1u);
LABEL_151:
          sub_1A153A0(v2, v91, v74);
          return 1;
        }
      }
    }
    else if ( v43 == 27 && (*(_DWORD *)(v42 + 20) & 0xFFFFFFFu) >> 1 != 1 )
    {
      v63 = (*(_BYTE *)(v42 + 23) & 0x40) != 0
          ? *(__int64 **)(v42 - 8)
          : (__int64 *)(v42 - 24LL * (*(_DWORD *)(v42 + 20) & 0xFFFFFFF));
      if ( (*(_BYTE *)sub_1A10F60(v2, *v63) & 6) == 0 )
      {
        if ( (*(_BYTE *)(v42 + 23) & 0x40) != 0 )
          v64 = *(_QWORD **)(v42 - 8);
        else
          v64 = (_QWORD *)(v42 - 24LL * (*(_DWORD *)(v42 + 20) & 0xFFFFFFF));
        if ( *(_BYTE *)(*v64 + 16LL) == 9 )
        {
          v80 = v64[1];
          v81 = v64[6];
          v82 = v64[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v82 = v80;
          if ( v80 )
            *(_QWORD *)(v80 + 16) = v82 | *(_QWORD *)(v80 + 16) & 3LL;
          *v64 = v81;
          if ( v81 )
          {
            v83 = *(_QWORD *)(v81 + 8);
            v64[1] = v83;
            if ( v83 )
              *(_QWORD *)(v83 + 16) = (unsigned __int64)(v64 + 1) | *(_QWORD *)(v83 + 16) & 3LL;
            v64[2] = (v81 + 8) | v64[2] & 3LL;
            *(_QWORD *)(v81 + 8) = v64;
          }
          if ( (*(_BYTE *)(v42 + 23) & 0x40) != 0 )
            v84 = *(_QWORD *)(v42 - 8);
          else
            v84 = v42 - 24LL * (*(_DWORD *)(v42 + 20) & 0xFFFFFFF);
          sub_1A153A0(v2, v91, *(_QWORD *)(v84 + 72));
          return 1;
        }
        v44 = v64[9];
        goto LABEL_69;
      }
    }
LABEL_6:
    v92 = *(_QWORD *)(v92 + 8);
    if ( v89 == v92 )
      return 0;
  }
  v6 = v2;
  while ( 1 )
  {
    if ( !v5 )
      BUG();
    v9 = *(_QWORD *)(v5 - 24);
    v10 = v5 - 24;
    v11 = *(_BYTE *)(v9 + 8);
    if ( !v11 )
      goto LABEL_14;
    if ( v11 != 13 )
      break;
    v7 = *(_BYTE *)(v5 - 8);
    if ( v7 <= 0x17u )
      goto LABEL_13;
    v8 = v10 | 4;
    if ( v7 != 78 )
    {
      v8 = v10 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v7 != 29 )
        goto LABEL_13;
    }
    v20 = v8 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_31;
    v21 = (__int64 *)(v20 - 24);
    v22 = (__int64 *)(v20 - 72);
    if ( (v8 & 4) != 0 )
      v22 = v21;
    if ( !*(_BYTE *)(*v22 + 16) )
    {
      if ( !sub_186B010(v6 + 312, *v22) && (unsigned __int8)(*(_BYTE *)(v5 - 8) - 86) > 1u )
      {
LABEL_31:
        v23 = *(_DWORD *)(v9 + 12);
        if ( v23 )
        {
          for ( i = 0; i != v23; ++i )
          {
            v25 = sub_1A11440(v6, v5 - 24, i);
            if ( (*v25 & 6) == 0 )
            {
              *v25 |= 6uLL;
              v28 = *(unsigned int *)(v6 + 824);
              if ( (unsigned int)v28 >= *(_DWORD *)(v6 + 828) )
              {
                sub_16CD150(v6 + 816, v88, 0, 8, v26, v27);
                v28 = *(unsigned int *)(v6 + 824);
              }
              *(_QWORD *)(*(_QWORD *)(v6 + 816) + 8 * v28) = v10;
              ++*(_DWORD *)(v6 + 824);
            }
          }
        }
      }
    }
    else
    {
LABEL_13:
      if ( (unsigned __int8)(v7 - 86) > 1u )
        goto LABEL_31;
    }
LABEL_14:
    v5 = *(_QWORD *)(v5 + 8);
    if ( v91 + 40 == v5 )
    {
      v2 = v6;
      goto LABEL_64;
    }
  }
  v12 = sub_1A10F60(v6, v5 - 24);
  v13 = (*v12 >> 1) & 3;
  if ( ((*v12 >> 1) & 3) != 0 || *(_BYTE *)(v5 - 8) == 86 )
    goto LABEL_14;
  if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
    v14 = *(__int64 **)(v5 - 32);
  else
    v14 = (__int64 *)(v10 - 24LL * (*(_DWORD *)(v5 - 4) & 0xFFFFFFF));
  v15 = *v14;
  if ( *(_BYTE *)(*(_QWORD *)*v14 + 8LL) == 13 )
    goto LABEL_54;
  v17 = *sub_1A10F60(v6, v15);
  if ( (*(_DWORD *)(v5 - 4) & 0xFFFFFFF) == 2 )
  {
    v29 = v5 - 72;
    if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
      v29 = *(_QWORD *)(v5 - 32);
    v15 = *(_QWORD *)(v29 + 24);
    if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 13 )
      goto LABEL_54;
    v13 = *sub_1A10F60(v6, v15);
  }
  v18 = *(__int64 ***)(v5 - 24);
  v19 = *(unsigned __int8 *)(v5 - 8);
  switch ( *(_BYTE *)(v5 - 8) )
  {
    case 0x1D:
    case 0x4E:
      if ( (unsigned __int8)v19 <= 0x17u )
      {
        v49 = 0;
LABEL_91:
        v50 = v49 - 72;
        goto LABEL_92;
      }
      if ( (_BYTE)v19 == 78 )
      {
        v65 = v10 | 4;
      }
      else
      {
        v49 = 0;
        if ( (_BYTE)v19 != 29 )
          goto LABEL_91;
        v65 = v10 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v49 = v65 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v65 & 4) == 0 )
        goto LABEL_91;
      v50 = v49 - 24;
LABEL_92:
      v51 = *(_QWORD *)v50;
      if ( *(_BYTE *)(*(_QWORD *)v50 + 16LL) )
        goto LABEL_54;
      v52 = *(_DWORD *)(v6 + 272);
      if ( !v52 )
        goto LABEL_54;
      v53 = v52 - 1;
      v54 = *(_QWORD *)(v6 + 256);
      v55 = (v52 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v56 = *(_QWORD *)(v54 + 16LL * v55);
      if ( v51 == v56 )
        goto LABEL_14;
      v57 = 1;
      while ( v56 != -8 )
      {
        v55 = v53 & (v57 + v55);
        v56 = *(_QWORD *)(v54 + 16LL * v55);
        if ( v51 == v56 )
          goto LABEL_14;
        ++v57;
      }
LABEL_54:
      v30 = v5 - 24;
      v31 = v6;
      goto LABEL_55;
    case 0x23:
    case 0x25:
    case 0x36:
    case 0x3C:
    case 0x43:
    case 0x47:
      goto LABEL_14;
    case 0x24:
    case 0x26:
    case 0x28:
    case 0x2B:
    case 0x2E:
      v16 = v17;
      v30 = v5 - 24;
      v31 = v6;
      if ( (((unsigned __int8)v16 | (unsigned __int8)v13) & 6) == 0 )
        goto LABEL_45;
      goto LABEL_55;
    case 0x27:
    case 0x32:
      if ( (((unsigned __int8)v17 | (unsigned __int8)v13) & 6) != 0 )
        goto LABEL_44;
      goto LABEL_14;
    case 0x29:
    case 0x2A:
    case 0x2C:
    case 0x2D:
      if ( ((v13 >> 1) & 3) != 0 )
      {
        if ( ((v13 >> 1) & 3) == 3 )
          goto LABEL_44;
        v86 = *(__int64 ***)(v5 - 24);
        v45 = sub_1595F50(v13 & 0xFFFFFFFFFFFFFFF8LL, v15, v19, v16);
        v18 = v86;
        if ( !v45 )
          goto LABEL_44;
      }
      goto LABEL_14;
    case 0x2F:
    case 0x30:
    case 0x31:
      if ( ((v13 >> 1) & 3) == 0 )
        goto LABEL_14;
      if ( ((v13 >> 1) & 3) == 3 )
        goto LABEL_44;
      v38 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_BYTE *)(v38 + 16) != 13 )
        goto LABEL_44;
      v39 = *(_DWORD *)(v38 + 32);
      if ( v39 > 0x40 )
      {
        v87 = *(__int64 ***)(v5 - 24);
        if ( v39 - (unsigned int)sub_16A57B0(v38 + 24) > 0x40 )
          goto LABEL_14;
        v18 = v87;
        v40 = **(_QWORD **)(v38 + 24);
      }
      else
      {
        v40 = *(_QWORD *)(v38 + 24);
      }
      v85 = v18;
      v41 = sub_16431D0(*(_QWORD *)v38);
      v18 = v85;
      if ( v41 > v40 )
      {
LABEL_44:
        v30 = v5 - 24;
        v31 = v6;
LABEL_45:
        v32 = sub_15A06D0(v18, v15, v19, v16);
        sub_1A108C0(v31, v30, v32);
        return 1;
      }
      goto LABEL_14;
    case 0x33:
      if ( (((unsigned __int8)v17 | (unsigned __int8)v13) & 6) == 0 )
        goto LABEL_14;
      v62 = sub_15A04A0(*(_QWORD ***)(v5 - 24));
      sub_1A108C0(v6, v5 - 24, v62);
      return 1;
    case 0x34:
      if ( (((unsigned __int8)v17 | (unsigned __int8)v13) & 6) == 0 )
        goto LABEL_44;
      goto LABEL_14;
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x44:
    case 0x45:
    case 0x46:
      goto LABEL_44;
    case 0x4B:
      if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
        v33 = *(__int64 **)(v5 - 32);
      else
        v33 = (__int64 *)(v10 - 24LL * (*(_DWORD *)(v5 - 4) & 0xFFFFFFF));
      v34 = *sub_1A10F60(v6, *v33);
      if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
        v35 = *(_QWORD *)(v5 - 32);
      else
        v35 = v10 - 24LL * (*(_DWORD *)(v5 - 4) & 0xFFFFFFF);
      v36 = *sub_1A10F60(v6, *(_QWORD *)(v35 + 24));
      if ( (v34 & 6) != 0 && (v36 & 6) != 0 )
        goto LABEL_54;
      v37 = *(unsigned __int16 *)(v5 - 6);
      BYTE1(v37) &= ~0x80u;
      if ( (unsigned int)(v37 - 32) > 1 )
        goto LABEL_54;
      goto LABEL_14;
    case 0x4F:
      if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
        v58 = *(_QWORD *)(v5 - 32);
      else
        v58 = v10 - 24LL * (*(_DWORD *)(v5 - 4) & 0xFFFFFFF);
      v59 = *sub_1A10F60(v6, *(_QWORD *)(v58 + 24));
      if ( (v17 & 6) != 0 )
      {
        v60 = (v59 >> 1) & 3;
        if ( ((v59 >> 1) & 3) != 0
          || ((*(_BYTE *)(v5 - 1) & 0x40) == 0
            ? (v61 = v10 - 24LL * (*(_DWORD *)(v5 - 4) & 0xFFFFFFF))
            : (v61 = *(_QWORD *)(v5 - 32)),
              v59 = *sub_1A10F60(v6, *(_QWORD *)(v61 + 48)),
              v60 = (v59 >> 1) & 3,
              (_DWORD)v60) )
        {
          v30 = v5 - 24;
          v31 = v6;
          goto LABEL_107;
        }
        goto LABEL_14;
      }
      v30 = v5 - 24;
      v31 = v6;
      v66 = (v59 >> 1) & 3;
      if ( v66 == 1 || v66 == 2 )
        goto LABEL_109;
      v67 = (*(_BYTE *)(v5 - 1) & 0x40) != 0 ? *(_QWORD *)(v5 - 32) : v30 - 24LL * (*(_DWORD *)(v5 - 4) & 0xFFFFFFF);
      v59 = *sub_1A10F60(v6, *(_QWORD *)(v67 + 48));
      v60 = (v59 >> 1) & 3;
LABEL_107:
      if ( v60 == 1 || v60 == 2 )
LABEL_109:
        sub_1A108C0(v31, v30, v59 & 0xFFFFFFFFFFFFFFF8LL);
      else
LABEL_55:
        sub_1A11830(v31, v30);
      return 1;
    default:
      goto LABEL_54;
  }
}
