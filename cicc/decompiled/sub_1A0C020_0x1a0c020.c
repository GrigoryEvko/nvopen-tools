// Function: sub_1A0C020
// Address: 0x1a0c020
//
void __fastcall sub_1A0C020(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128 a4,
        __m128 a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  int v10; // edx
  _QWORD *v12; // rbx
  __int64 v13; // r12
  unsigned int v14; // ecx
  __int64 v15; // rdi
  char v16; // al
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  char v20; // al
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v23; // rdi
  _QWORD *v24; // rax
  char v25; // bl
  char v26; // dl
  __int64 *v27; // rax
  __int64 *v28; // rdi
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // rdx
  __int64 **v32; // rax
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 *v35; // rdx
  __int64 v36; // rsi
  unsigned __int64 v37; // rcx
  __int64 v38; // rcx
  double v39; // xmm4_8
  double v40; // xmm5_8
  __int64 v41; // rsi
  __int64 *v42; // r8
  bool v43; // al
  bool v44; // dl
  unsigned __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rax
  double v49; // xmm4_8
  double v50; // xmm5_8
  __int64 v51; // rdi
  __int64 v52; // r15
  __int64 v53; // rax
  __int64 v54; // rbx
  _QWORD *v55; // rax
  __int64 v56; // rax
  double v57; // xmm4_8
  double v58; // xmm5_8
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // rbx
  _QWORD *v62; // rax
  double v63; // xmm4_8
  double v64; // xmm5_8
  __int64 v65; // rax
  __int64 v66; // rbx
  __int64 v67; // rdi
  _QWORD *v68; // rax
  _QWORD *v69; // rax
  __int64 v70; // rsi
  unsigned __int8 *v71; // rsi
  int v72; // eax
  _QWORD *v73; // rax
  _QWORD *v74; // rax
  bool v75; // [rsp-60h] [rbp-60h]
  unsigned int v76; // [rsp-60h] [rbp-60h]
  bool v77; // [rsp-60h] [rbp-60h]
  __int64 i; // [rsp-60h] [rbp-60h]
  __int64 *v79; // [rsp-60h] [rbp-60h]
  __int64 v80[2]; // [rsp-58h] [rbp-58h] BYREF
  __int16 v81; // [rsp-48h] [rbp-48h]

  v10 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned int)(v10 - 35) > 0x11 )
    return;
  v12 = (_QWORD *)a2;
  if ( (_BYTE)v10 == 47 )
  {
    v27 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
        ? *(__int64 **)(a2 - 8)
        : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(v27[3] + 16) == 13 )
    {
      if ( sub_19FF050(*v27, 15)
        || (v67 = *(_QWORD *)(a2 + 8)) != 0
        && !*(_QWORD *)(v67 + 8)
        && ((v68 = sub_1648700(v67), sub_19FF050((__int64)v68, 15))
         || (v69 = sub_1648700(*(_QWORD *)(a2 + 8)), sub_19FF050((__int64)v69, 11))) )
      {
        v28 = (__int64 *)sub_15A0680(*(_QWORD *)a2, 1, 0);
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v29 = *(_QWORD *)(a2 - 8);
        else
          v29 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v30 = *(_QWORD *)(v29 + 24);
        v31 = sub_15A2D50(v28, v30, 0, 0, *(double *)a3.m128_u64, *(double *)a4.m128_u64, *(double *)a5.m128_u64);
        v81 = 257;
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v32 = *(__int64 ***)(a2 - 8);
        else
          v32 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        v33 = sub_15FB440(15, *v32, v31, (__int64)v80, a2);
        v34 = sub_1599EF0(*(__int64 ***)a2);
        if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
          v35 = *(__int64 **)(a2 - 8);
        else
          v35 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        if ( *v35 )
        {
          v36 = v35[1];
          v37 = v35[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v37 = v36;
          if ( v36 )
            *(_QWORD *)(v36 + 16) = *(_QWORD *)(v36 + 16) & 3LL | v37;
        }
        *v35 = v34;
        if ( v34 )
        {
          v38 = *(_QWORD *)(v34 + 8);
          v35[1] = v38;
          if ( v38 )
            *(_QWORD *)(v38 + 16) = (unsigned __int64)(v35 + 1) | *(_QWORD *)(v38 + 16) & 3LL;
          v35[2] = (v34 + 8) | v35[2] & 3;
          *(_QWORD *)(v34 + 8) = v35;
        }
        sub_164B7C0(v33, (__int64)v12);
        sub_164D160(
          (__int64)v12,
          v33,
          a3,
          *(double *)a4.m128_u64,
          *(double *)a5.m128_u64,
          *(double *)a6.m128i_i64,
          v39,
          v40,
          a9,
          a10);
        v41 = v12[6];
        v42 = (__int64 *)(v33 + 48);
        v80[0] = v41;
        if ( v41 )
        {
          sub_1623A60((__int64)v80, v41, 2);
          v42 = (__int64 *)(v33 + 48);
          if ( (__int64 *)(v33 + 48) == v80 )
          {
            if ( v80[0] )
              sub_161E7C0((__int64)v80, v80[0]);
            goto LABEL_45;
          }
          v70 = *(_QWORD *)(v33 + 48);
          if ( !v70 )
          {
LABEL_102:
            v71 = (unsigned __int8 *)v80[0];
            *(_QWORD *)(v33 + 48) = v80[0];
            if ( v71 )
              sub_1623210((__int64)v80, v71, (__int64)v42);
            goto LABEL_45;
          }
        }
        else if ( v42 == v80 || (v70 = *(_QWORD *)(v33 + 48)) == 0 )
        {
LABEL_45:
          v75 = sub_15F2380((__int64)v12);
          v43 = sub_15F2370((__int64)v12);
          v44 = v43;
          if ( v75 )
          {
            if ( v43 )
              goto LABEL_50;
            v76 = *(_DWORD *)(v30 + 32);
            if ( v76 > 0x40 )
            {
              v72 = sub_16A57B0(v30 + 24);
              v44 = 0;
              if ( v76 - v72 > 0x40 )
                goto LABEL_51;
              v45 = **(_QWORD **)(v30 + 24);
            }
            else
            {
              v45 = *(_QWORD *)(v30 + 24);
            }
            if ( (unsigned int)((*(_DWORD *)(*v12 + 8LL) >> 8) - 1) > v45 )
            {
LABEL_50:
              v77 = v44;
              sub_15F2330(v33, 1);
              v44 = v77;
            }
          }
LABEL_51:
          sub_15F2310(v33, v44);
          v80[0] = (__int64)v12;
          sub_1A062A0(a1 + 64, v80);
          v12 = (_QWORD *)v33;
          *(_BYTE *)(a1 + 752) = 1;
          goto LABEL_3;
        }
        v79 = v42;
        sub_161E7C0((__int64)v42, v70);
        v42 = v79;
        goto LABEL_102;
      }
    }
  }
LABEL_3:
  v13 = sub_1A06D50(
          a1,
          (__int64)v12,
          a3,
          *(double *)a4.m128_u64,
          *(double *)a5.m128_u64,
          *(double *)a6.m128i_i64,
          a7,
          a8,
          a9,
          a10);
  if ( !v13 )
    v13 = (__int64)v12;
  v14 = *(unsigned __int8 *)(v13 + 16) - 24;
  if ( v14 > 0x1C || ((1LL << v14) & 0x1C019800) == 0 )
  {
    v15 = *(_QWORD *)v13;
    v16 = *(_BYTE *)(*(_QWORD *)v13 + 8LL);
    if ( v16 != 16 )
      goto LABEL_8;
    goto LABEL_23;
  }
  sub_1A04030(a1, v13);
  v15 = *(_QWORD *)v13;
  v16 = *(_BYTE *)(*(_QWORD *)v13 + 8LL);
  if ( v16 == 16 )
LABEL_23:
    v16 = *(_BYTE *)(**(_QWORD **)(v15 + 16) + 8LL);
LABEL_8:
  if ( (unsigned __int8)(v16 - 1) <= 5u )
  {
    if ( !sub_15F2480(v13) )
      return;
    v15 = *(_QWORD *)v13;
  }
  v17 = 1;
  if ( sub_1642F90(v15, 1) )
    return;
  v20 = *(_BYTE *)(v13 + 16);
  if ( v20 == 37 )
  {
    if ( sub_15FB6B0(v13, 1, v18, v19) || (v17 = 0, sub_15FB6D0(v13, 0, v46, v47)) || !sub_1A014D0(v13) )
    {
      if ( !sub_15FB6B0(v13, v17, v46, v47) )
        goto LABEL_14;
      v48 = (*(_BYTE *)(v13 + 23) & 0x40) != 0 ? *(_QWORD *)(v13 - 8) : v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
      if ( !sub_19FF050(*(_QWORD *)(v48 + 24), 15) )
        goto LABEL_14;
      v51 = *(_QWORD *)(v13 + 8);
      if ( v51 )
      {
        if ( !*(_QWORD *)(v51 + 8) )
        {
          v73 = sub_1648700(v51);
          if ( sub_19FF050((__int64)v73, 15) )
            goto LABEL_14;
        }
      }
      v52 = a1 + 64;
      v53 = sub_19FFCB0(
              (__int64 ***)v13,
              a3,
              *(double *)a4.m128_u64,
              *(double *)a5.m128_u64,
              *(double *)a6.m128i_i64,
              v49,
              v50,
              a9,
              a10);
      v54 = *(_QWORD *)(v53 + 8);
      for ( i = v53; v54; v54 = *(_QWORD *)(v54 + 8) )
      {
        v55 = sub_1648700(v54);
        if ( (unsigned __int8)(*((_BYTE *)v55 + 16) - 35) <= 0x11u )
        {
          v80[0] = (__int64)v55;
          sub_1A062A0(a1 + 64, v80);
        }
      }
LABEL_67:
      v80[0] = v13;
      sub_1A062A0(v52, v80);
      *(_BYTE *)(a1 + 752) = 1;
      v13 = i;
      goto LABEL_14;
    }
LABEL_82:
    v65 = sub_1A06A90(
            v13,
            a1 + 64,
            a3,
            *(double *)a4.m128_u64,
            *(double *)a5.m128_u64,
            *(double *)a6.m128i_i64,
            v63,
            v64,
            a9,
            a10,
            v46,
            v47);
    v80[0] = v13;
    v66 = v65;
    sub_1A062A0(a1 + 64, v80);
    v13 = v66;
    *(_BYTE *)(a1 + 752) = 1;
    goto LABEL_14;
  }
  if ( v20 == 38 )
  {
    if ( sub_15FB6B0(v13, 1, v18, v19) || sub_15FB6D0(v13, 0, v46, v47) || !sub_1A014D0(v13) )
    {
      if ( !sub_15FB6D0(v13, 0, v46, v47) )
        goto LABEL_14;
      v56 = (*(_BYTE *)(v13 + 23) & 0x40) != 0 ? *(_QWORD *)(v13 - 8) : v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
      if ( !sub_19FF050(*(_QWORD *)(v56 + 24), 16) )
        goto LABEL_14;
      v59 = *(_QWORD *)(v13 + 8);
      if ( v59 )
      {
        if ( !*(_QWORD *)(v59 + 8) )
        {
          v74 = sub_1648700(v59);
          if ( sub_19FF050((__int64)v74, 16) )
            goto LABEL_14;
        }
      }
      v52 = a1 + 64;
      v60 = sub_19FFCB0(
              (__int64 ***)v13,
              a3,
              *(double *)a4.m128_u64,
              *(double *)a5.m128_u64,
              *(double *)a6.m128i_i64,
              v57,
              v58,
              a9,
              a10);
      v61 = *(_QWORD *)(v60 + 8);
      for ( i = v60; v61; v61 = *(_QWORD *)(v61 + 8) )
      {
        v62 = sub_1648700(v61);
        if ( (unsigned __int8)(*((_BYTE *)v62 + 16) - 35) <= 0x11u )
        {
          v80[0] = (__int64)v62;
          sub_1A062A0(a1 + 64, v80);
        }
      }
      goto LABEL_67;
    }
    goto LABEL_82;
  }
LABEL_14:
  if ( !(unsigned __int8)sub_15F34B0(v13) )
    return;
  v23 = *(_QWORD *)(v13 + 8);
  if ( !v23 || *(_QWORD *)(v23 + 8) )
    goto LABEL_20;
  v24 = sub_1648700(v23);
  v25 = *(_BYTE *)(v13 + 16);
  v26 = *((_BYTE *)v24 + 16);
  if ( v25 != v26 )
  {
    if ( v25 == 35 )
    {
      if ( v26 == 37 )
        return;
    }
    else if ( v25 == 36 && v26 == 38 )
    {
      return;
    }
LABEL_20:
    sub_1A0B880(a1, v13, a3, a4, a5, a6, v21, v22, a9, a10);
    return;
  }
  if ( v24 != (_QWORD *)v13 && v24[5] == *(_QWORD *)(v13 + 40) )
  {
    v80[0] = (__int64)v24;
    sub_1A062A0(a1 + 64, v80);
  }
}
