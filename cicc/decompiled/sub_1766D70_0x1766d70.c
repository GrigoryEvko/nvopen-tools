// Function: sub_1766D70
// Address: 0x1766d70
//
__int64 __fastcall sub_1766D70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 **v16; // rdx
  __int64 **v17; // rax
  __int64 *v18; // r15
  __int64 v19; // r13
  unsigned int v20; // ebx
  unsigned int v21; // eax
  __int64 **v22; // rdx
  unsigned int v23; // r12d
  __int64 i; // r13
  _QWORD *v25; // rax
  unsigned __int8 v26; // dl
  _QWORD *v27; // r12
  int v29; // eax
  __int64 v30; // rax
  __int64 *v31; // rdi
  unsigned int v32; // eax
  __int64 *v33; // rcx
  const char *v34; // rsi
  __int64 v35; // rsi
  unsigned __int8 *v36; // rsi
  __int64 *v37; // rax
  __int64 v38; // rax
  unsigned int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rdx
  unsigned __int64 v42; // rax
  __int64 v43; // rax
  unsigned __int8 v44; // al
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rdi
  unsigned int v48; // edx
  unsigned __int64 v49; // rax
  unsigned int v50; // edx
  __int64 v51; // rax
  __int64 *v52; // rax
  unsigned __int8 *v53; // rax
  __int64 v54; // rbx
  __int64 *v55; // r12
  __int64 v56; // rax
  unsigned __int8 *v57; // rax
  double v58; // xmm4_8
  double v59; // xmm5_8
  __int64 v60; // rdi
  __int64 ***v61; // rax
  __int64 ***v62; // r15
  double v63; // xmm4_8
  double v64; // xmm5_8
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 *v67; // [rsp+10h] [rbp-B0h]
  __int64 *v68; // [rsp+18h] [rbp-A8h]
  unsigned int v69; // [rsp+18h] [rbp-A8h]
  __int64 v71; // [rsp+30h] [rbp-90h]
  unsigned int v72; // [rsp+30h] [rbp-90h]
  __int64 *v73; // [rsp+38h] [rbp-88h]
  unsigned __int8 *v74; // [rsp+38h] [rbp-88h]
  __int64 **v77; // [rsp+58h] [rbp-68h] BYREF
  unsigned __int64 v78; // [rsp+60h] [rbp-60h] BYREF
  __int64 *v79; // [rsp+68h] [rbp-58h]
  __int64 v80; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v81; // [rsp+78h] [rbp-48h]
  __int16 v82; // [rsp+80h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v13 = *(__int64 **)(a2 - 8);
  else
    v13 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v14 = *v13;
  v15 = v13[3];
  if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
    v16 = *(__int64 ***)(v14 - 8);
  else
    v16 = (__int64 **)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
  v73 = *v16;
  if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
    v17 = *(__int64 ***)(v15 - 8);
  else
    v17 = (__int64 **)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
  v18 = *v17;
  v19 = **v17;
  v71 = *v73;
  v20 = sub_1643030(*v73);
  v21 = sub_1643030(v19);
  v22 = (__int64 **)v71;
  v23 = v21;
  if ( v20 >= v21 )
    v21 = v20;
  else
    v22 = (__int64 **)v19;
  v77 = v22;
  v72 = v21;
  if ( (unsigned __int8)sub_1648D00(a2, 2) )
  {
    for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
    {
      v25 = sub_1648700(i);
      if ( v25 != (_QWORD *)a1 )
      {
        v26 = *((_BYTE *)v25 + 16);
        if ( v26 <= 0x17u )
          return 0;
        if ( v26 == 60 )
        {
          if ( (unsigned int)sub_1643030(*v25) > v72 )
            return 0;
          continue;
        }
        if ( v26 != 50 )
          return 0;
        v47 = *(v25 - 3);
        if ( *(_BYTE *)(v47 + 16) != 13 )
          return 0;
        v48 = *(_DWORD *)(v47 + 32);
        if ( v48 > 0x40 )
        {
          v50 = v48 - sub_16A57B0(v47 + 24);
        }
        else
        {
          v49 = *(_QWORD *)(v47 + 24);
          if ( !v49 )
            continue;
          _BitScanReverse64(&v49, v49);
          v50 = 64 - (v49 ^ 0x3F);
        }
        if ( v72 < v50 )
          return 0;
      }
    }
  }
  v29 = *(unsigned __int16 *)(a1 + 18);
  BYTE1(v29) &= ~0x80u;
  switch ( v29 )
  {
    case ' ':
    case '!':
      v44 = *(_BYTE *)(a3 + 16);
      if ( v44 <= 0x17u )
      {
        if ( v44 != 5 )
          return 0;
        if ( *(_WORD *)(a3 + 18) != 26 )
          return 0;
        v45 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
        if ( !v45 )
          return 0;
        v46 = *(_QWORD *)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(v46 + 16) != 13 )
          return 0;
LABEL_65:
        if ( a2 == v45 )
        {
          sub_13A38D0((__int64)&v80, v46 + 24);
          sub_16A7490((__int64)&v80, 1);
          v69 = v81;
          LODWORD(v79) = v81;
          v78 = v80;
          if ( sub_14A9C60((__int64)&v78) && v72 == v69 - 1 - (unsigned int)sub_1455840((__int64)&v78) )
          {
LABEL_37:
            sub_135E100((__int64 *)&v78);
            goto LABEL_38;
          }
          sub_135E100((__int64 *)&v78);
        }
        return 0;
      }
      if ( v44 != 61 )
      {
        if ( v44 != 50 )
          return 0;
        v45 = *(_QWORD *)(a3 - 48);
        if ( !v45 )
          return 0;
        v46 = *(_QWORD *)(a3 - 24);
        if ( *(_BYTE *)(v46 + 16) != 13 )
          return 0;
        goto LABEL_65;
      }
      v65 = *(_QWORD *)(a3 + 8);
      if ( !v65 )
        return 0;
      if ( *(_QWORD *)(v65 + 8) )
        return 0;
      v66 = *(_QWORD *)(a3 - 24);
      if ( *(_BYTE *)(v66 + 16) != 60 || v72 != (unsigned int)sub_1643030(*(_QWORD *)v66) )
        return 0;
LABEL_38:
      v33 = (__int64 *)a4[1];
      v33[1] = *(_QWORD *)(a2 + 40);
      v33[2] = a2 + 24;
      v34 = *(const char **)(a2 + 48);
      v68 = v33;
      v80 = (__int64)v34;
      if ( v34 )
      {
        sub_1623A60((__int64)&v80, (__int64)v34, 2);
        v35 = *v68;
        if ( !*v68 )
          goto LABEL_41;
      }
      else
      {
        v35 = *v33;
        if ( !*v33 )
          goto LABEL_43;
      }
      sub_161E7C0((__int64)v68, v35);
LABEL_41:
      v36 = (unsigned __int8 *)v80;
      *v68 = v80;
      if ( v36 )
      {
        sub_1623210((__int64)&v80, v36, (__int64)v68);
      }
      else if ( v80 )
      {
        sub_161E7C0((__int64)&v80, v80);
      }
LABEL_43:
      if ( v20 < v72 )
      {
        v82 = 257;
        v73 = (__int64 *)sub_1708970((__int64)v68, 37, (__int64)v73, v77, &v80);
      }
      if ( v23 < v72 )
      {
        v82 = 257;
        v18 = (__int64 *)sub_1708970((__int64)v68, 37, (__int64)v18, v77, &v80);
      }
      v37 = (__int64 *)sub_15F2050(a1);
      v38 = sub_15E26F0(v37, 210, (__int64 *)&v77, 1);
      v82 = 259;
      v80 = (__int64)"umul";
      v79 = v18;
      v78 = (unsigned __int64)v73;
      v67 = (__int64 *)sub_172C570((__int64)v68, *(_QWORD *)(*(_QWORD *)v38 + 24LL), v38, (__int64 *)&v78, 2, &v80, 0);
      sub_170B990(*a4, a2);
      if ( (unsigned __int8)sub_1648D00(a2, 2) )
      {
        v80 = (__int64)"umul.value";
        v82 = 259;
        LODWORD(v78) = 0;
        v53 = sub_1759FE0((__int64)v68, (__int64)v67, (unsigned int *)&v78, 1, &v80);
        v54 = *(_QWORD *)(a2 + 8);
        v55 = (__int64 *)v53;
        while ( v54 )
        {
          v60 = v54;
          v54 = *(_QWORD *)(v54 + 8);
          v61 = (__int64 ***)sub_1648700(v60);
          v62 = v61;
          if ( (__int64 ***)a3 != v61 && (__int64 ***)a1 != v61 )
          {
            if ( *((_BYTE *)v61 + 16) == 60 )
            {
              if ( v72 == (unsigned int)sub_1643030((__int64)*v61) )
                sub_170E100(a4, (__int64)v62, (__int64)v55, a5, a6, a7, a8, v63, v64, a11, a12);
              else
                sub_1593B40(v62 - 3, (__int64)v55);
            }
            else
            {
              sub_16A5A50((__int64)&v78, (__int64 *)*(v61 - 3) + 3, v72);
              v82 = 257;
              v56 = sub_15A1070(*v55, (__int64)&v78);
              v57 = sub_1729500((__int64)v68, (unsigned __int8 *)v55, v56, &v80, *(double *)a5.m128_u64, a6, a7);
              v82 = 257;
              v74 = sub_1708970((__int64)v68, 37, (__int64)v57, *v62, &v80);
              sub_170B990(*a4, (__int64)v74);
              sub_170E100(a4, (__int64)v62, (__int64)v74, a5, a6, a7, a8, v58, v59, a11, a12);
              if ( (unsigned int)v79 > 0x40 && v78 )
                j_j___libc_free_0_0(v78);
            }
            sub_170B990(*a4, (__int64)v62);
          }
        }
      }
      if ( *(_BYTE *)(a3 + 16) > 0x17u )
        sub_170B990(*a4, a3);
      v39 = *(unsigned __int16 *)(a1 + 18);
      BYTE1(v39) &= ~0x80u;
      if ( v39 > 0x23 )
      {
        v51 = *(_QWORD *)(a1 - 24);
        if ( a2 != v51 )
          goto LABEL_86;
      }
      else
      {
        if ( v39 <= 0x21 )
        {
          if ( v39 != 32 )
            goto LABEL_53;
LABEL_86:
          v82 = 257;
          LODWORD(v78) = 1;
          v52 = (__int64 *)sub_1759FE0((__int64)v68, (__int64)v67, (unsigned int *)&v78, 1, &v80);
          v82 = 257;
          return sub_15FB630(v52, (__int64)&v80, 0);
        }
        v51 = *(_QWORD *)(a1 - 48);
        if ( a2 != v51 )
          goto LABEL_86;
      }
      if ( !v51 )
        goto LABEL_86;
LABEL_53:
      LODWORD(v78) = 1;
      v82 = 257;
      v27 = sub_1648A60(88, 1u);
      if ( v27 )
      {
        v40 = sub_15FB2A0(*v67, (unsigned int *)&v78, 1);
        sub_15F1EA0((__int64)v27, v40, 62, (__int64)(v27 - 3), 1, 0);
        if ( *(v27 - 3) )
        {
          v41 = *(v27 - 2);
          v42 = *(v27 - 1) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v42 = v41;
          if ( v41 )
            *(_QWORD *)(v41 + 16) = *(_QWORD *)(v41 + 16) & 3LL | v42;
        }
        *(v27 - 3) = v67;
        v43 = v67[1];
        *(v27 - 2) = v43;
        if ( v43 )
          *(_QWORD *)(v43 + 16) = (unsigned __int64)(v27 - 2) | *(_QWORD *)(v43 + 16) & 3LL;
        *(v27 - 1) = (unsigned __int64)(v67 + 1) | *(v27 - 1) & 3LL;
        v67[1] = (__int64)(v27 - 3);
        v27[7] = v27 + 9;
        v27[8] = 0x400000000LL;
        sub_15FB110((__int64)v27, &v78, 1, (__int64)&v80);
      }
      return (__int64)v27;
    case '"':
    case '%':
      if ( *(_BYTE *)(a3 + 16) != 13 )
        return 0;
      LODWORD(v79) = v72;
      if ( v72 <= 0x40 )
        v78 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v72;
      else
        sub_16A4EF0((__int64)&v78, -1, 1);
      sub_16A5C50((__int64)&v80, (const void **)&v78, *(_DWORD *)(a3 + 32));
      if ( (unsigned int)v79 > 0x40 && v78 )
        j_j___libc_free_0_0(v78);
      v78 = v80;
      v32 = v81;
      v81 = 0;
      LODWORD(v79) = v32;
      sub_135E100(&v80);
      v31 = (__int64 *)&v78;
      if ( sub_1455820((__int64)&v78, (_QWORD *)(a3 + 24)) )
        goto LABEL_37;
      goto LABEL_91;
    case '#':
    case '$':
      if ( *(_BYTE *)(a3 + 16) != 13 )
        return 0;
      v81 = *(_DWORD *)(a3 + 32);
      v30 = 1LL << v72;
      if ( v81 <= 0x40 )
      {
        v80 = 0;
      }
      else
      {
        sub_16A4EF0((__int64)&v80, 0, 0);
        v30 = 1LL << v72;
        if ( v81 > 0x40 )
        {
          *(_QWORD *)(v80 + 8LL * (v72 >> 6)) |= 1LL << v72;
          goto LABEL_28;
        }
      }
      v80 |= v30;
LABEL_28:
      v31 = &v80;
      if ( sub_1455820((__int64)&v80, (_QWORD *)(a3 + 24)) )
      {
        sub_135E100(&v80);
        goto LABEL_38;
      }
LABEL_91:
      sub_135E100(v31);
      return 0;
    default:
      return 0;
  }
}
