// Function: sub_3874770
// Address: 0x3874770
//
__int64 __fastcall sub_3874770(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        double a7,
        double a8,
        double a9)
{
  _QWORD *v13; // rcx
  _QWORD *v14; // rax
  int v15; // r15d
  __int64 v16; // rsi
  _QWORD *v17; // rbx
  unsigned __int64 v18; // rax
  int v19; // edx
  _QWORD *v20; // rdi
  bool v21; // al
  bool v22; // al
  bool v23; // al
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  int v31; // eax
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // rcx
  __int64 v35; // rsi
  int v36; // edx
  unsigned int v37; // edi
  __int64 *v38; // rax
  __int64 v39; // r8
  __int64 v40; // r15
  __int64 v41; // rdi
  unsigned __int64 v42; // rax
  __int64 v43; // rsi
  bool v44; // cc
  __int64 v45; // r8
  __int64 *v46; // r13
  int v47; // eax
  int v48; // r9d
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r8
  char v52; // al
  __int64 v53; // rdx
  int v54; // r13d
  __int64 v55; // rdi
  __int64 *v56; // r13
  __int64 v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rsi
  unsigned __int8 *v60; // rsi
  __int64 v61; // [rsp+0h] [rbp-D0h]
  _QWORD *v62; // [rsp+8h] [rbp-C8h]
  __int64 v64; // [rsp+10h] [rbp-C0h]
  __int64 v65; // [rsp+10h] [rbp-C0h]
  __int64 v66; // [rsp+10h] [rbp-C0h]
  __int64 v67; // [rsp+10h] [rbp-C0h]
  __int64 v68; // [rsp+10h] [rbp-C0h]
  __int64 v69; // [rsp+10h] [rbp-C0h]
  __int64 v70; // [rsp+10h] [rbp-C0h]
  _QWORD *v71; // [rsp+18h] [rbp-B8h]
  __int64 v72; // [rsp+18h] [rbp-B8h]
  __int64 v73; // [rsp+18h] [rbp-B8h]
  __int64 v74; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v75; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v76; // [rsp+40h] [rbp-90h]
  __int64 v77[2]; // [rsp+50h] [rbp-80h] BYREF
  __int16 v78; // [rsp+60h] [rbp-70h]
  _QWORD v79[3]; // [rsp+70h] [rbp-60h] BYREF
  _QWORD v80[9]; // [rsp+88h] [rbp-48h] BYREF

  if ( *(_BYTE *)(a3 + 16) <= 0x10u && *(_BYTE *)(a4 + 16) <= 0x10u )
    return sub_15A2A30((__int64 *)a2, (__int64 *)a3, a4, 0, 0, a7, a8, a9);
  v13 = *(_QWORD **)(*(_QWORD *)(a1 + 272) + 48LL);
  v14 = *(_QWORD **)(a1 + 280);
  if ( v13 != v14 )
  {
    v15 = 6;
    v16 = 0x80A800000000LL;
    v17 = (_QWORD *)(*v14 & 0xFFFFFFFFFFFFFFF8LL);
    while ( 1 )
    {
      if ( !v17 )
        BUG();
      v18 = *((unsigned __int8 *)v17 - 8);
      if ( (_BYTE)v18 == 78 )
      {
        v24 = *(v17 - 6);
        if ( !*(_BYTE *)(v24 + 16) && (*(_BYTE *)(v24 + 33) & 0x20) != 0 )
          v15 += (unsigned int)(*(_DWORD *)(v24 + 36) - 35) < 4;
      }
      v19 = (unsigned __int8)v18;
      if ( a2 == (unsigned __int8)v18 - 24 )
      {
        if ( (*((_BYTE *)v17 - 1) & 0x40) != 0 )
        {
          v20 = (_QWORD *)*(v17 - 4);
        }
        else
        {
          a5 = 24LL * (*((_DWORD *)v17 - 1) & 0xFFFFFFF);
          v20 = (_QWORD *)((char *)v17 - a5 - 24);
        }
        if ( a3 == *v20 && a4 == v20[3] )
        {
          a5 = (__int64)(v17 - 3);
          if ( (unsigned __int8)v18 > 0x2Fu || !_bittest64(&v16, v18) )
            goto LABEL_20;
          v71 = v13;
          v21 = sub_15F2380((__int64)(v17 - 3));
          v13 = v71;
          v16 = 0x80A800000000LL;
          if ( !v21 )
          {
            v22 = sub_15F2370((__int64)(v17 - 3));
            v13 = v71;
            v16 = 0x80A800000000LL;
            if ( !v22 )
              break;
          }
        }
      }
LABEL_6:
      if ( v13 != v17 )
      {
        v17 = (_QWORD *)(*v17 & 0xFFFFFFFFFFFFFFF8LL);
        if ( --v15 )
          continue;
      }
      v14 = *(_QWORD **)(a1 + 280);
      goto LABEL_29;
    }
    v19 = *((unsigned __int8 *)v17 - 8);
    a5 = (__int64)(v17 - 3);
    LOBYTE(v18) = *((_BYTE *)v17 - 8);
LABEL_20:
    if ( (unsigned int)(v19 - 41) > 1 && (unsigned __int8)(v18 - 48) > 1u )
      return a5;
    v62 = v13;
    v72 = a5;
    v23 = sub_15F23D0(a5);
    a5 = v72;
    v13 = v62;
    v16 = 0x80A800000000LL;
    if ( !v23 )
      return a5;
    goto LABEL_6;
  }
LABEL_29:
  if ( !v14 )
    BUG();
  v25 = v14[3];
  v74 = v25;
  if ( v25 )
  {
    sub_1623A60((__int64)&v74, v25, 2);
    v14 = *(_QWORD **)(a1 + 280);
  }
  v26 = *(_QWORD *)(a1 + 272);
  v27 = *(_QWORD *)(a1 + 264);
  v79[2] = v14;
  v79[0] = a1 + 264;
  v79[1] = v26;
  v80[0] = v27;
  if ( v27 )
    sub_1623A60((__int64)v80, v27, 2);
  v80[1] = a1;
  v28 = *(unsigned int *)(a1 + 344);
  if ( (unsigned int)v28 >= *(_DWORD *)(a1 + 348) )
  {
    sub_16CD150(a1 + 336, (const void *)(a1 + 352), 0, 8, a5, a6);
    v28 = *(unsigned int *)(a1 + 344);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 336) + 8 * v28) = v79;
  v29 = *(_QWORD *)a1;
  ++*(_DWORD *)(a1 + 344);
  v30 = *(_QWORD *)(v29 + 64);
  v31 = *(_DWORD *)(v30 + 24);
  if ( !v31 )
    goto LABEL_51;
  while ( 1 )
  {
    v34 = *(_QWORD *)(a1 + 272);
    v35 = *(_QWORD *)(v30 + 8);
    v36 = v31 - 1;
    v37 = (v31 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
    v38 = (__int64 *)(v35 + 16LL * v37);
    v39 = *v38;
    if ( *v38 != v34 )
      break;
LABEL_44:
    v40 = v38[1];
    if ( !v40 )
      goto LABEL_51;
    if ( !sub_13FC1A0(v38[1], a3) )
      goto LABEL_51;
    if ( !sub_13FC1A0(v40, a4) )
      goto LABEL_51;
    v41 = sub_13FC520(v40);
    if ( !v41 )
      goto LABEL_51;
    v42 = sub_157EBA0(v41);
    *(_QWORD *)(a1 + 272) = *(_QWORD *)(v42 + 40);
    *(_QWORD *)(a1 + 280) = v42 + 24;
    v43 = *(_QWORD *)(v42 + 48);
    v77[0] = v43;
    if ( v43 )
    {
      sub_1623A60((__int64)v77, v43, 2);
      v32 = *(_QWORD *)(a1 + 264);
      if ( v32 )
        goto LABEL_39;
LABEL_40:
      v33 = (unsigned __int8 *)v77[0];
      *(_QWORD *)(a1 + 264) = v77[0];
      if ( v33 )
        sub_1623210((__int64)v77, v33, a1 + 264);
      v30 = *(_QWORD *)(*(_QWORD *)a1 + 64LL);
      v31 = *(_DWORD *)(v30 + 24);
      if ( !v31 )
        goto LABEL_51;
    }
    else
    {
      v32 = *(_QWORD *)(a1 + 264);
      if ( v32 )
      {
LABEL_39:
        sub_161E7C0(a1 + 264, v32);
        goto LABEL_40;
      }
      v30 = *(_QWORD *)(*(_QWORD *)a1 + 64LL);
      v31 = *(_DWORD *)(v30 + 24);
      if ( !v31 )
        goto LABEL_51;
    }
  }
  v47 = 1;
  while ( v39 != -8 )
  {
    v48 = v47 + 1;
    v37 = v36 & (v47 + v37);
    v38 = (__int64 *)(v35 + 16LL * v37);
    v39 = *v38;
    if ( *v38 == v34 )
      goto LABEL_44;
    v47 = v48;
  }
LABEL_51:
  v44 = *(_BYTE *)(a3 + 16) <= 0x10u;
  v76 = 257;
  if ( !v44
    || *(_BYTE *)(a4 + 16) > 0x10u
    || (v61 = sub_15A2A30((__int64 *)a2, (__int64 *)a3, a4, 0, 0, a7, a8, a9),
        (v45 = sub_14DBA30(v61, *(_QWORD *)(a1 + 328), 0)) == 0)
    && (v45 = v61) == 0 )
  {
    v78 = 257;
    v49 = sub_15FB440(a2, (__int64 *)a3, a4, (__int64)v77, 0);
    v50 = *(_QWORD *)v49;
    v51 = v49;
    v52 = *(_BYTE *)(*(_QWORD *)v49 + 8LL);
    if ( v52 == 16 )
      v52 = *(_BYTE *)(**(_QWORD **)(v50 + 16) + 8LL);
    if ( (unsigned __int8)(v52 - 1) <= 5u || *(_BYTE *)(v51 + 16) == 76 )
    {
      v53 = *(_QWORD *)(a1 + 296);
      v54 = *(_DWORD *)(a1 + 304);
      if ( v53 )
      {
        v65 = v51;
        sub_1625C10(v51, 3, v53);
        v51 = v65;
      }
      v66 = v51;
      sub_15F2440(v51, v54);
      v51 = v66;
    }
    v55 = *(_QWORD *)(a1 + 272);
    if ( v55 )
    {
      v56 = *(__int64 **)(a1 + 280);
      v67 = v51;
      sub_157E9D0(v55 + 40, v51);
      v51 = v67;
      v57 = *v56;
      v58 = *(_QWORD *)(v67 + 24);
      *(_QWORD *)(v67 + 32) = v56;
      v57 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v67 + 24) = v57 | v58 & 7;
      *(_QWORD *)(v57 + 8) = v67 + 24;
      *v56 = *v56 & 7 | (v67 + 24);
    }
    v68 = v51;
    sub_164B780(v51, &v75);
    sub_12A86E0((__int64 *)(a1 + 264), v68);
    v45 = v68;
  }
  v46 = (__int64 *)(v45 + 48);
  v77[0] = v74;
  if ( !v74 )
  {
    if ( v46 == v77 )
      goto LABEL_58;
    v59 = *(_QWORD *)(v45 + 48);
    if ( !v59 )
      goto LABEL_58;
LABEL_77:
    v69 = v45;
    sub_161E7C0((__int64)v46, v59);
    v45 = v69;
    goto LABEL_78;
  }
  v64 = v45;
  sub_1623A60((__int64)v77, v74, 2);
  v45 = v64;
  if ( v46 == v77 )
  {
    if ( v77[0] )
    {
      sub_161E7C0((__int64)v77, v77[0]);
      v45 = v64;
    }
    goto LABEL_58;
  }
  v59 = *(_QWORD *)(v64 + 48);
  if ( v59 )
    goto LABEL_77;
LABEL_78:
  v60 = (unsigned __int8 *)v77[0];
  *(_QWORD *)(v45 + 48) = v77[0];
  if ( v60 )
  {
    v70 = v45;
    sub_1623210((__int64)v77, v60, (__int64)v46);
    v45 = v70;
  }
LABEL_58:
  v73 = v45;
  sub_38740E0(a1, v45);
  sub_3870260((__int64)v79);
  a5 = v73;
  if ( v74 )
  {
    sub_161E7C0((__int64)&v74, v74);
    return v73;
  }
  return a5;
}
