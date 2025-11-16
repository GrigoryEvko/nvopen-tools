// Function: sub_17617A0
// Address: 0x17617a0
//
__int64 __fastcall sub_17617A0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 *v15; // rbx
  unsigned int v16; // r15d
  __int64 v17; // rcx
  unsigned __int8 v18; // al
  unsigned int v19; // r12d
  bool v20; // al
  unsigned int v21; // eax
  unsigned int v22; // ecx
  unsigned __int64 v23; // r15
  signed int v24; // r12d
  bool v25; // r13
  __int16 v26; // r13
  __int64 v27; // r14
  int v28; // eax
  _QWORD *v29; // r12
  _QWORD **v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rsi
  bool v36; // dl
  unsigned __int64 v38; // rax
  unsigned int v39; // ecx
  unsigned __int64 v40; // r12
  int v41; // eax
  _QWORD **v42; // rax
  _QWORD *v43; // r15
  __int64 *v44; // rax
  __int64 v45; // rsi
  unsigned __int64 v46; // rax
  int v47; // eax
  int v48; // r12d
  int v49; // eax
  unsigned __int64 v50; // rax
  int v51; // eax
  __int64 v52; // rax
  double v53; // xmm4_8
  double v54; // xmm5_8
  unsigned int v55; // eax
  unsigned int v56; // r10d
  __int128 v57; // rax
  unsigned int v58; // edx
  bool v59; // al
  char v60; // cl
  unsigned __int64 v61; // rax
  bool v62; // al
  int v63; // eax
  unsigned int v64; // [rsp+8h] [rbp-68h]
  unsigned __int8 v65; // [rsp+Ch] [rbp-64h]
  char v66; // [rsp+Ch] [rbp-64h]
  unsigned int v67; // [rsp+Ch] [rbp-64h]
  __int64 v68; // [rsp+18h] [rbp-58h]
  _QWORD *v69; // [rsp+18h] [rbp-58h]
  unsigned __int64 v70; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v71; // [rsp+28h] [rbp-48h]
  __int16 v72; // [rsp+30h] [rbp-40h]

  v15 = (__int64 *)a3;
  v16 = *(_DWORD *)(a5 + 8);
  v68 = a2;
  if ( v16 <= 0x40 )
    LOBYTE(a3) = *(_QWORD *)a5 == 0;
  else
    LOBYTE(a3) = v16 == (unsigned int)sub_16A57B0(a5);
  if ( (_BYTE)a3 )
    return 0;
  v17 = *(_QWORD *)(a2 - 48);
  v18 = *(_BYTE *)(v17 + 16);
  if ( v18 <= 0x17u )
  {
    if ( v18 != 5 || *(_WORD *)(v17 + 18) != 25 )
      goto LABEL_6;
  }
  else if ( v18 != 49 )
  {
LABEL_6:
    v19 = *(_DWORD *)(a4 + 8);
    goto LABEL_7;
  }
  if ( v16 > 0x40 )
  {
    if ( v16 != (unsigned int)sub_16A58F0(a5) )
    {
      v33 = 1LL << ((unsigned __int8)v16 - 1);
      v34 = *(_QWORD *)(*(_QWORD *)a5 + 8LL * ((v16 - 1) >> 6));
      goto LABEL_33;
    }
    return 0;
  }
  if ( *(_QWORD *)a5 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v16) )
    return 0;
  v34 = *(_QWORD *)a5;
  v33 = 1LL << ((unsigned __int8)v16 - 1);
LABEL_33:
  v19 = *(_DWORD *)(a4 + 8);
  v35 = *(_QWORD *)a4;
  v36 = (v34 & v33) != 0;
  if ( v19 > 0x40 )
    v35 = *(_QWORD *)(v35 + 8LL * ((v19 - 1) >> 6));
  if ( ((v35 & (1LL << ((unsigned __int8)v19 - 1))) != 0) != v36 )
    return 0;
  a2 = a4;
  if ( (int)sub_16AEA10(a5, a4) > 0 )
    return 0;
  a3 = 1;
LABEL_7:
  if ( v19 > 0x40 )
  {
    v65 = a3;
    if ( (unsigned int)sub_16A57B0(a4) != v19 )
    {
      a2 = a5;
      v20 = sub_16A5220(a4, (const void **)a5);
      a3 = v65;
      if ( !v20 )
      {
        if ( v65 && (*(_QWORD *)(*(_QWORD *)a4 + 8LL * ((v19 - 1) >> 6)) & (1LL << ((unsigned __int8)v19 - 1))) != 0 )
        {
          v48 = sub_16A5810(a4);
          goto LABEL_63;
        }
        v21 = sub_16A57B0(a4);
        LOBYTE(a3) = v65;
        v19 = v21;
        goto LABEL_12;
      }
      goto LABEL_47;
    }
LABEL_53:
    if ( v16 > 0x40 )
    {
      v45 = v16 - 1 - (unsigned int)sub_16A57B0(a5);
    }
    else
    {
      v45 = 0xFFFFFFFFLL;
      if ( *(_QWORD *)a5 )
      {
        _BitScanReverse64(&v46, *(_QWORD *)a5);
        v45 = 63 - ((unsigned int)v46 ^ 0x3F);
      }
    }
    v26 = 34;
    v27 = sub_15A0680(*v15, v45, 0);
    v47 = *(unsigned __int16 *)(v68 + 18);
    BYTE1(v47) &= ~0x80u;
    if ( v47 == 33 )
      v26 = sub_15FF0F0(34);
LABEL_49:
    v72 = 257;
    v29 = sub_1648A60(56, 2u);
    if ( v29 )
    {
      v42 = (_QWORD **)*v15;
      if ( *(_BYTE *)(*v15 + 8) == 16 )
      {
        v43 = v42[4];
        v44 = (__int64 *)sub_1643320(*v42);
        v32 = (__int64)sub_16463B0(v44, (unsigned int)v43);
      }
      else
      {
        v32 = sub_1643320(*v42);
      }
      goto LABEL_52;
    }
    return (__int64)v29;
  }
  if ( !*(_QWORD *)a4 )
    goto LABEL_53;
  v38 = *(_QWORD *)a4;
  if ( *(_QWORD *)a4 == *(_QWORD *)a5 )
  {
LABEL_47:
    v26 = 32;
    v27 = sub_15A06D0((__int64 **)*v15, a2, a3, v17);
    v41 = *(unsigned __int16 *)(v68 + 18);
    BYTE1(v41) &= ~0x80u;
    if ( v41 == 33 )
      v26 = sub_15FF0F0(32);
    goto LABEL_49;
  }
  if ( (_BYTE)a3 && _bittest64((const __int64 *)&v38, v19 - 1) )
  {
    v60 = 64 - v19;
    v48 = 64;
    v61 = ~(v38 << v60);
    if ( v61 )
    {
      _BitScanReverse64(&v61, v61);
      v48 = v61 ^ 0x3F;
    }
LABEL_63:
    if ( v16 > 0x40 )
    {
      v49 = sub_16A5810(a5);
    }
    else
    {
      v49 = 64;
      if ( *(_QWORD *)a5 << (64 - (unsigned __int8)v16) != -1 )
      {
        _BitScanReverse64(&v50, ~(*(_QWORD *)a5 << (64 - (unsigned __int8)v16)));
        v49 = v50 ^ 0x3F;
      }
    }
    v24 = v48 - v49;
    if ( v24 <= 0 )
      goto LABEL_67;
LABEL_70:
    sub_13A38D0((__int64)&v70, a5);
    v56 = v71;
    if ( v71 > 0x40 )
    {
      sub_16A5E70((__int64)&v70, v24);
      v56 = v71;
    }
    else
    {
      v57 = (__int64)(v70 << (64 - (unsigned __int8)v71)) >> (64 - (unsigned __int8)v71);
      *(_QWORD *)&v57 = (__int64)v57 >> v24;
      if ( v71 == v24 )
        *(_QWORD *)&v57 = *((_QWORD *)&v57 + 1);
      v70 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v71) & v57;
    }
    v58 = *(_DWORD *)(a4 + 8);
    if ( v58 <= 0x40 )
    {
      if ( *(_QWORD *)a4 != v70 )
        goto LABEL_76;
    }
    else
    {
      v64 = *(_DWORD *)(a4 + 8);
      v67 = v56;
      v59 = sub_16A5220(a4, (const void **)&v70);
      v56 = v67;
      v58 = v64;
      if ( !v59 )
      {
LABEL_76:
        if ( v56 > 0x40 && v70 )
          j_j___libc_free_0_0(v70);
        goto LABEL_17;
      }
    }
    if ( v56 > 0x40 && v70 )
    {
      j_j___libc_free_0_0(v70);
      v58 = *(_DWORD *)(a4 + 8);
    }
    if ( v58 <= 0x40 )
      v62 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v58) == *(_QWORD *)a4;
    else
      v62 = v58 == (unsigned int)sub_16A58F0(a4);
    if ( !v62 )
      goto LABEL_23;
    if ( *(_DWORD *)(a5 + 8) > 0x40u )
    {
      if ( (unsigned int)sub_16A5940(a5) != 1 )
        goto LABEL_93;
    }
    else if ( !*(_QWORD *)a5 || (*(_QWORD *)a5 & (*(_QWORD *)a5 - 1LL)) != 0 )
    {
LABEL_93:
      v26 = 35;
      v27 = sub_15A0680(*v15, v24, 0);
      v63 = *(unsigned __int16 *)(v68 + 18);
      BYTE1(v63) &= ~0x80u;
      if ( v63 == 33 )
        v26 = sub_15FF0F0(35);
LABEL_25:
      v72 = 257;
      v29 = sub_1648A60(56, 2u);
      if ( v29 )
      {
        v30 = (_QWORD **)*v15;
        if ( *(_BYTE *)(*v15 + 8) == 16 )
        {
          v69 = v30[4];
          v31 = (__int64 *)sub_1643320(*v30);
          v32 = (__int64)sub_16463B0(v31, (unsigned int)v69);
        }
        else
        {
          v32 = sub_1643320(*v30);
        }
LABEL_52:
        sub_15FEC10((__int64)v29, v32, 51, v26, (__int64)v15, v27, (__int64)&v70, 0);
        return (__int64)v29;
      }
      return (__int64)v29;
    }
LABEL_23:
    v26 = 32;
    v27 = sub_15A0680(*v15, v24, 0);
    v28 = *(unsigned __int16 *)(v68 + 18);
    BYTE1(v28) &= ~0x80u;
    if ( v28 == 33 )
      v26 = sub_15FF0F0(32);
    goto LABEL_25;
  }
  v39 = v19 - 64;
  if ( v38 )
  {
    _BitScanReverse64(&v40, v38);
    v19 = v39 + (v40 ^ 0x3F);
  }
LABEL_12:
  if ( v16 > 0x40 )
  {
    v66 = a3;
    v55 = sub_16A57B0(a5);
    LOBYTE(a3) = v66;
    v16 = v55;
  }
  else
  {
    v22 = v16 - 64;
    if ( *(_QWORD *)a5 )
    {
      _BitScanReverse64(&v23, *(_QWORD *)a5);
      v16 = v22 + (v23 ^ 0x3F);
    }
  }
  v24 = v19 - v16;
  if ( v24 <= 0 )
    goto LABEL_67;
  if ( (_BYTE)a3 )
    goto LABEL_70;
LABEL_17:
  sub_13A38D0((__int64)&v70, a5);
  if ( v71 > 0x40 )
  {
    sub_16A8110((__int64)&v70, v24);
  }
  else if ( v24 == v71 )
  {
    v70 = 0;
  }
  else
  {
    v70 >>= v24;
  }
  if ( *(_DWORD *)(a4 + 8) <= 0x40u )
    v25 = *(_QWORD *)a4 == v70;
  else
    v25 = sub_16A5220(a4, (const void **)&v70);
  sub_135E100((__int64 *)&v70);
  if ( v25 )
    goto LABEL_23;
LABEL_67:
  v51 = *(unsigned __int16 *)(v68 + 18);
  BYTE1(v51) &= ~0x80u;
  v52 = sub_15A0680(*(_QWORD *)v68, v51 == 33, 0);
  return sub_170E100(a1, v68, v52, a6, a7, a8, a9, v53, v54, a12, a13);
}
