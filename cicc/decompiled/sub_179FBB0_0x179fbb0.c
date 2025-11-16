// Function: sub_179FBB0
// Address: 0x179fbb0
//
__int64 __fastcall sub_179FBB0(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  __int64 v12; // rbx
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 v15; // rcx
  double v16; // xmm4_8
  double v17; // xmm5_8
  unsigned __int8 v18; // al
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 *v22; // r10
  __int64 v23; // r11
  char v24; // al
  __int64 v25; // r11
  __int64 *v26; // rdi
  int v27; // esi
  __int64 v28; // rdi
  __int64 *v29; // rax
  __int64 v30; // rax
  char v31; // al
  unsigned __int8 *v32; // r15
  __int64 v33; // r8
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // rdx
  unsigned int v40; // eax
  __int64 v41; // rdi
  __int64 v42; // rax
  unsigned __int8 *v43; // r12
  __int64 v44; // rdx
  unsigned __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  int v52; // eax
  int v53; // eax
  unsigned __int8 v54; // [rsp+Fh] [rbp-A1h]
  __int64 *v55; // [rsp+10h] [rbp-A0h]
  __int64 v56; // [rsp+10h] [rbp-A0h]
  __int64 v57; // [rsp+10h] [rbp-A0h]
  __int64 v58; // [rsp+10h] [rbp-A0h]
  __int64 *v59; // [rsp+18h] [rbp-98h]
  __int64 v60; // [rsp+18h] [rbp-98h]
  __int64 v61; // [rsp+18h] [rbp-98h]
  __int64 v62; // [rsp+18h] [rbp-98h]
  __int64 v63; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v64; // [rsp+28h] [rbp-88h]
  __int64 v65; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v66; // [rsp+38h] [rbp-78h]
  __int64 v67[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v68; // [rsp+50h] [rbp-60h]
  __int64 v69[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v70; // [rsp+70h] [rbp-40h]

  v10 = a2;
  v12 = a2;
  v13 = *(_QWORD *)(a2 - 48);
  v14 = *(_QWORD *)(a2 - 24);
  if ( (unsigned __int8)sub_17AD890(a1, a2) )
    return v10;
  v18 = *(_BYTE *)(v14 + 16);
  if ( *(_BYTE *)(v13 + 16) <= 0x10u )
  {
    if ( v18 == 79 )
    {
      v19 = sub_1707470((__int64)a1, (_BYTE *)a2, v14, *(double *)a3.m128_u64, a4, a5);
      if ( v19 )
        return v19;
      if ( *(_BYTE *)(v14 + 16) > 0x10u )
        goto LABEL_19;
    }
    else if ( v18 > 0x10u )
    {
      goto LABEL_9;
    }
  }
  else if ( v18 > 0x10u )
  {
    goto LABEL_20;
  }
  a2 = v13;
  v19 = sub_179E920(a1, v13, (char *)v14, v12, a3, a4, a5, a6, v16, v17, a9, a10);
  if ( v19 )
    return v19;
LABEL_19:
  if ( *(_BYTE *)(v13 + 16) > 0x10u )
    goto LABEL_20;
  v18 = *(_BYTE *)(v14 + 16);
LABEL_9:
  if ( v18 == 35 )
  {
    v22 = *(__int64 **)(v14 - 48);
    if ( !v22 )
      goto LABEL_20;
    v23 = *(_QWORD *)(v14 - 24);
    if ( *(_BYTE *)(v23 + 16) > 0x10u )
      goto LABEL_20;
  }
  else
  {
    if ( v18 != 5 )
      goto LABEL_20;
    if ( *(_WORD *)(v14 + 18) != 11 )
      goto LABEL_20;
    v21 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
    v15 = 4 * v21;
    v22 = *(__int64 **)(v14 - 24 * v21);
    if ( !v22 )
      goto LABEL_20;
    v23 = *(_QWORD *)(v14 + 24 * (1 - v21));
    if ( !v23 )
      goto LABEL_20;
  }
  a2 = a1[333];
  v55 = (__int64 *)v23;
  v59 = v22;
  v24 = sub_14C2730(v22, a2, 0, a1[330], v12, a1[332]);
  v25 = (__int64)v55;
  if ( v24 )
  {
    v26 = v55;
    a2 = a1[333];
    v56 = (__int64)v59;
    v60 = v25;
    if ( (unsigned __int8)sub_14C2730(v26, a2, 0, a1[330], v12, a1[332]) )
    {
      v27 = *(unsigned __int8 *)(v12 + 16);
      v28 = a1[1];
      v70 = 257;
      v68 = 257;
      v29 = (__int64 *)sub_17066B0(v28, v27 - 24, v13, v60, v67, 0, *(double *)a3.m128_u64, a4, a5);
      return sub_15FB440((unsigned int)*(unsigned __int8 *)(v12 + 16) - 24, v29, v56, (__int64)v69, 0);
    }
  }
LABEL_20:
  v30 = *(_QWORD *)(v14 + 8);
  if ( !v30 || *(_QWORD *)(v30 + 8) )
    return 0;
  v31 = *(_BYTE *)(v14 + 16);
  if ( v31 == 45 )
  {
    v32 = *(unsigned __int8 **)(v14 - 48);
    if ( !v32 )
      return 0;
    v33 = *(_QWORD *)(v14 - 24);
    v34 = *(unsigned __int8 *)(v33 + 16);
    if ( (_BYTE)v34 == 13 )
    {
      a2 = v33 + 24;
      if ( *(_DWORD *)(v33 + 32) > 0x40u )
      {
        v54 = *(_BYTE *)(v33 + 16);
        v57 = *(_QWORD *)(v14 - 24);
        v61 = v33 + 24;
        v52 = sub_16A5940(v33 + 24);
        a2 = v61;
        v33 = v57;
        v34 = v54;
        if ( v52 == 1 )
          goto LABEL_38;
      }
      else
      {
        v35 = *(_QWORD *)(v33 + 24);
        if ( v35 )
        {
          v15 = v35 - 1;
          if ( (v35 & (v35 - 1)) == 0 )
            goto LABEL_38;
        }
      }
    }
    if ( *(_BYTE *)(*(_QWORD *)v33 + 8LL) != 16 || (unsigned __int8)v34 > 0x10u )
      return 0;
LABEL_33:
    v36 = sub_15A1020((_BYTE *)v33, a2, v34, v15);
    if ( v36 && *(_BYTE *)(v36 + 16) == 13 )
    {
      a2 = v36 + 24;
      if ( *(_DWORD *)(v36 + 32) > 0x40u )
      {
        a2 = v36 + 24;
        if ( (unsigned int)sub_16A5940(v36 + 24) == 1 )
          goto LABEL_38;
      }
      else
      {
        v37 = *(_QWORD *)(v36 + 24);
        if ( v37 && (v37 & (v37 - 1)) == 0 )
          goto LABEL_38;
      }
    }
    return 0;
  }
  if ( v31 != 5 )
    return 0;
  if ( *(_WORD *)(v14 + 18) != 21 )
    return 0;
  v49 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
  v15 = 4 * v49;
  v32 = *(unsigned __int8 **)(v14 - 24 * v49);
  if ( !v32 )
    return 0;
  v34 = 1 - v49;
  v50 = 3 * (1 - v49);
  v33 = *(_QWORD *)(v14 + 8 * v50);
  if ( *(_BYTE *)(v33 + 16) != 13 )
  {
LABEL_59:
    if ( *(_BYTE *)(*(_QWORD *)v33 + 8LL) != 16 )
      return 0;
    goto LABEL_33;
  }
  a2 = v33 + 24;
  if ( *(_DWORD *)(v33 + 32) > 0x40u )
  {
    v58 = *(_QWORD *)(v14 + 8 * v50);
    v62 = v33 + 24;
    v53 = sub_16A5940(v33 + 24);
    a2 = v62;
    v33 = v58;
    if ( v53 == 1 )
      goto LABEL_38;
    goto LABEL_59;
  }
  v51 = *(_QWORD *)(v33 + 24);
  if ( !v51 )
    goto LABEL_59;
  v34 = v51 - 1;
  if ( (v51 & (v51 - 1)) != 0 )
    goto LABEL_59;
LABEL_38:
  v38 = a1[1];
  v67[0] = (__int64)sub_1649960(v14);
  v70 = 261;
  v67[1] = v39;
  v69[0] = (__int64)v67;
  v64 = *(_DWORD *)(a2 + 8);
  if ( v64 > 0x40 )
    sub_16A4FD0((__int64)&v63, (const void **)a2);
  else
    v63 = *(_QWORD *)a2;
  sub_16A7800((__int64)&v63, 1u);
  v40 = v64;
  v41 = *(_QWORD *)v12;
  v64 = 0;
  v66 = v40;
  v65 = v63;
  v42 = sub_15A1070(v41, (__int64)&v65);
  v43 = sub_1729500(v38, v32, v42, v69, *(double *)a3.m128_u64, a4, a5);
  if ( v66 > 0x40 && v65 )
    j_j___libc_free_0_0(v65);
  if ( v64 > 0x40 && v63 )
    j_j___libc_free_0_0(v63);
  if ( *(_QWORD *)(v12 - 24) )
  {
    v44 = *(_QWORD *)(v12 - 16);
    v45 = *(_QWORD *)(v12 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v45 = v44;
    if ( v44 )
      *(_QWORD *)(v44 + 16) = *(_QWORD *)(v44 + 16) & 3LL | v45;
  }
  *(_QWORD *)(v12 - 24) = v43;
  if ( v43 )
  {
    v46 = *((_QWORD *)v43 + 1);
    *(_QWORD *)(v12 - 16) = v46;
    if ( v46 )
      *(_QWORD *)(v46 + 16) = (v12 - 16) | *(_QWORD *)(v46 + 16) & 3LL;
    v47 = *(_QWORD *)(v12 - 8);
    v48 = v12 - 24;
    *(_QWORD *)(v48 + 16) = (unsigned __int64)(v43 + 8) | v47 & 3;
    *((_QWORD *)v43 + 1) = v48;
  }
  return v10;
}
