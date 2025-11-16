// Function: sub_D3C740
// Address: 0xd3c740
//
__int64 __fastcall sub_D3C740(__int64 *a1, __int64 *a2, unsigned int a3, __int64 *a4, unsigned int a5)
{
  __int64 result; // rax
  __int64 v7; // r12
  char v8; // bl
  __int64 v9; // r13
  __int64 v10; // rsi
  unsigned int v11; // edx
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rcx
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rax
  unsigned int v17; // eax
  unsigned __int64 v18; // rbx
  __int64 v19; // r15
  __int64 v20; // rax
  unsigned __int64 v21; // r13
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rsi
  const void **v36; // rsi
  int v37; // edx
  int v38; // ecx
  int v39; // eax
  unsigned __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned __int64 v44; // rcx
  unsigned __int64 v45; // rax
  bool v46; // r8
  unsigned __int64 v47; // rax
  __int64 v48; // rdi
  unsigned int v49; // edx
  __int64 v50; // rsi
  unsigned __int64 v51; // rax
  __int64 v52; // rcx
  unsigned __int64 v53; // rsi
  __int64 v54; // rsi
  unsigned int v55; // edx
  __int64 v56; // rax
  unsigned __int64 v57; // rcx
  __int64 v58; // [rsp+0h] [rbp-E0h]
  unsigned __int64 v59; // [rsp+8h] [rbp-D8h]
  __int64 v60; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v61; // [rsp+18h] [rbp-C8h]
  char v62; // [rsp+24h] [rbp-BCh]
  char v63; // [rsp+25h] [rbp-BBh]
  char v64; // [rsp+26h] [rbp-BAh]
  char v65; // [rsp+27h] [rbp-B9h]
  unsigned __int64 v66; // [rsp+28h] [rbp-B8h]
  unsigned __int64 *v67; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v68; // [rsp+38h] [rbp-A8h]
  __int64 v69; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v70; // [rsp+48h] [rbp-98h]
  __int64 v71; // [rsp+50h] [rbp-90h] BYREF
  __int64 v72; // [rsp+58h] [rbp-88h]
  __int16 v73; // [rsp+60h] [rbp-80h]
  __int64 v74; // [rsp+68h] [rbp-78h]
  __m128i v75; // [rsp+70h] [rbp-70h] BYREF
  unsigned __int64 v76; // [rsp+80h] [rbp-60h]
  char v77; // [rsp+88h] [rbp-58h]
  char v78; // [rsp+90h] [rbp-50h]
  unsigned __int64 v79; // [rsp+98h] [rbp-48h]
  char v80; // [rsp+A0h] [rbp-40h]
  char v81; // [rsp+A1h] [rbp-3Fh]
  char v82; // [rsp+A8h] [rbp-38h]

  sub_D3BE90(&v75, a1, a2, *(_QWORD *)(a1[7] + 8LL * a3), a4, *(_QWORD *)(a1[7] + 8LL * a5));
  if ( !v82 )
    return v75.m128i_u32[0];
  if ( v82 != 1 )
    abort();
  v7 = v75.m128i_i64[0];
  v8 = v78;
  v61 = v76;
  v62 = v78;
  v65 = v77;
  v63 = v80;
  v64 = v81;
  v66 = v79;
  if ( (unsigned __int8)sub_D96A50(v75.m128i_i64[0]) )
  {
    *((_BYTE *)a1 + 224) |= v8;
    return 1;
  }
  v9 = *(_QWORD *)(*a1 + 112);
  v60 = sub_AA4E30(**(_QWORD **)(a1[1] + 32));
  if ( v66 )
  {
    v22 = v75.m128i_i64[1];
    v58 = sub_DEF9D0(*a1, a1);
    v23 = sub_D95540(v58);
    v72 = sub_DA2C50(v9, v23, v22, 0);
    v69 = (__int64)&v71;
    v71 = v58;
    v70 = 0x200000002LL;
    v24 = sub_DC8BD0(v9, &v69, 0, 0);
    if ( (__int64 *)v69 != &v71 )
      _libc_free(v69, &v69);
    v25 = sub_D95540(v7);
    v26 = sub_9208B0(v60, v25);
    v70 = v27;
    v69 = v26;
    v59 = sub_CA1930(&v69);
    v28 = sub_D95540(v24);
    v69 = sub_9208B0(v60, v28);
    v70 = v29;
    if ( v59 > sub_CA1930(&v69) )
    {
      v31 = v7;
      v41 = sub_D95540(v7);
      v24 = sub_DC2B70(v9, v24, v41, 0);
    }
    else
    {
      v30 = sub_D95540(v24);
      v31 = sub_DD2D10(v9, v7, v30);
    }
    v32 = sub_DCC810(v9, v31, v24, 0, 0);
    if ( (unsigned __int8)sub_DBEDC0(v9, v32) )
      return 0;
    v33 = sub_DCAF50(v9, v31, 0);
    v34 = sub_DCC810(v9, v33, v24, 0, 0);
    if ( (unsigned __int8)sub_DBEDC0(v9, v34) )
      return 0;
  }
  if ( *(_WORD *)(v7 + 24) )
  {
    if ( !*((_BYTE *)a1 + 440) )
    {
      sub_DE4EA0(&v69, a1[1], v9);
      if ( *((_BYTE *)a1 + 440) )
      {
        v54 = *((unsigned int *)a1 + 104);
        *((_BYTE *)a1 + 440) = 0;
        sub_C7D6A0(a1[50], 16 * v54, 8);
      }
      v42 = v70;
      a1[49] = 1;
      a1[50] = v42;
      v43 = v71;
      *((_BYTE *)a1 + 440) = 1;
      a1[51] = v43;
      ++v69;
      *((_DWORD *)a1 + 104) = v72;
      v70 = 0;
      *((_WORD *)a1 + 212) = v73;
      v71 = 0;
      a1[54] = v74;
      LODWORD(v72) = 0;
      sub_C7D6A0(0, 0, 8);
    }
    v35 = v7;
    v7 = 0;
    v19 = sub_DE2740(v9, v35, a1 + 49);
    if ( !(unsigned __int8)sub_DBEC80(v9, v19) )
      goto LABEL_25;
LABEL_37:
    if ( (unsigned __int8)sub_DBED40(v9, v19) )
    {
      if ( !v66 )
        return 1;
      return 3;
    }
    if ( v64 == 1 || !v63 || !(_BYTE)qword_4F87068 )
      return 3;
    if ( !v7 )
      goto LABEL_70;
    if ( !v66 )
      return 4;
    v48 = *(_QWORD *)(v7 + 32);
    v49 = *(_DWORD *)(v48 + 32);
    v50 = 1LL << ((unsigned __int8)v49 - 1);
    v51 = *(_QWORD *)(v48 + 24);
    if ( v49 > 0x40 )
    {
      if ( (*(_QWORD *)(v51 + 8LL * ((v49 - 1) >> 6)) & v50) == 0 )
      {
        v68 = *(_DWORD *)(v48 + 32);
        sub_C43780((__int64)&v67, (const void **)(v48 + 24));
        v55 = v68;
        goto LABEL_101;
      }
      LODWORD(v70) = *(_DWORD *)(v48 + 32);
      sub_C43780((__int64)&v69, (const void **)(v48 + 24));
      v49 = v70;
      if ( (unsigned int)v70 > 0x40 )
      {
        sub_C43D10((__int64)&v69);
LABEL_107:
        sub_C46250((__int64)&v69);
        v55 = v70;
        v68 = v70;
        v67 = (unsigned __int64 *)v69;
LABEL_101:
        v51 = (unsigned __int64)v67;
        if ( v55 > 0x40 )
        {
          v53 = *v67;
          goto LABEL_92;
        }
LABEL_91:
        v53 = v51;
LABEL_92:
        if ( !sub_D35550((__int64)a1, v53, v66) )
        {
          if ( v68 > 0x40 && v67 )
            j_j___libc_free_0_0(v67);
          return 3;
        }
        if ( v68 > 0x40 && v67 )
          j_j___libc_free_0_0(v67);
        return 4;
      }
      v52 = v69;
    }
    else
    {
      v52 = *(_QWORD *)(v48 + 24);
      if ( (v50 & v51) == 0 )
      {
        v68 = *(_DWORD *)(v48 + 32);
        v67 = (unsigned __int64 *)v51;
        goto LABEL_91;
      }
      LODWORD(v70) = *(_DWORD *)(v48 + 32);
    }
    v56 = ~v52;
    v57 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v49;
    if ( !v49 )
      v57 = 0;
    v69 = v56 & v57;
    goto LABEL_107;
  }
  v10 = *(_QWORD *)(v7 + 32);
  v11 = *(_DWORD *)(v10 + 32);
  v12 = *(_QWORD *)(v10 + 24);
  v13 = 1LL << ((unsigned __int8)v11 - 1);
  if ( v11 <= 0x40 )
  {
    v14 = *(_QWORD *)(v10 + 24);
    if ( (v13 & v12) == 0 )
    {
      v67 = *(unsigned __int64 **)(v10 + 24);
      goto LABEL_60;
    }
    LODWORD(v70) = *(_DWORD *)(v10 + 32);
    goto LABEL_12;
  }
  v36 = (const void **)(v10 + 24);
  if ( (*(_QWORD *)(v12 + 8LL * ((v11 - 1) >> 6)) & v13) != 0 )
  {
    LODWORD(v70) = v11;
    sub_C43780((__int64)&v69, v36);
    v11 = v70;
    if ( (unsigned int)v70 > 0x40 )
    {
      sub_C43D10((__int64)&v69);
LABEL_15:
      sub_C46250((__int64)&v69);
      v17 = v70;
      v68 = v70;
      v67 = (unsigned __int64 *)v69;
      goto LABEL_16;
    }
    v14 = v69;
LABEL_12:
    v15 = ~v14 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v11);
    v16 = 0;
    if ( v11 )
      v16 = v15;
    v69 = v16;
    goto LABEL_15;
  }
  v68 = v11;
  sub_C43780((__int64)&v67, v36);
  v17 = v68;
LABEL_16:
  if ( v17 <= 0x40 )
  {
LABEL_60:
    v18 = (unsigned __int64)v67;
    goto LABEL_18;
  }
  v18 = *v67;
  j_j___libc_free_0_0(v67);
LABEL_18:
  if ( v18 && v65 && v66 && v61 > 1 && !(v18 % v66) && v18 % v61 )
    return 0;
  v19 = v7;
  if ( (unsigned __int8)sub_DBEC80(v9, v7) )
    goto LABEL_37;
LABEL_25:
  v20 = sub_DBB9F0(v9, v19, 1, 0);
  sub_AB14C0((__int64)&v69, v20);
  if ( (unsigned int)v70 > 0x40 )
  {
    v21 = *(_QWORD *)v69;
    j_j___libc_free_0_0(v69);
  }
  else
  {
    if ( !(_DWORD)v70 )
    {
LABEL_70:
      *((_BYTE *)a1 + 224) |= v62;
      return 1;
    }
    v21 = v69 << (64 - (unsigned __int8)v70) >> (64 - (unsigned __int8)v70);
  }
  if ( (__int64)v21 <= 0 )
    goto LABEL_70;
  if ( !v7 )
    *((_BYTE *)a1 + 224) |= v62;
  if ( !v66 || !v65 )
    return 1;
  v37 = 1;
  v38 = 1;
  if ( dword_4F87428[0] )
    v38 = dword_4F87428[0];
  if ( dword_4F87508[0] )
    v37 = dword_4F87508[0];
  v39 = v37 * v38;
  if ( (unsigned int)(v37 * v38) < 2 )
    v39 = 2;
  v40 = v66 + v61 * (unsigned int)(v39 - 1);
  if ( v21 >= v40 )
  {
    v44 = a1[26];
    result = 5;
    if ( v44 < v40 )
      return result;
    v45 = a1[26];
    if ( v21 <= v44 )
      v45 = v21;
    a1[26] = v45;
    if ( v63 != 1 && v64 && (_BYTE)qword_4F87068 )
    {
      if ( v7 )
      {
        v46 = sub_D35550((__int64)a1, v21, v66);
        result = 7;
        if ( v46 )
          return result;
        v47 = 8 * v66 * (a1[26] / v61);
        goto LABEL_82;
      }
      v47 = 8 * v66 * (v45 / v61);
    }
    else
    {
      v47 = 8 * v66 * (v45 / v61);
      if ( v7 )
      {
LABEL_82:
        if ( a1[27] <= v47 )
          v47 = a1[27];
        a1[27] = v47;
        return 6;
      }
    }
    if ( *((unsigned int *)a1 + 88) > v47 )
      return 1;
    goto LABEL_82;
  }
  result = 5;
  if ( !v7 )
    return 1;
  return result;
}
