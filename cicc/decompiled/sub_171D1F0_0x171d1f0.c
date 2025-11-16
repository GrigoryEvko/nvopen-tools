// Function: sub_171D1F0
// Address: 0x171d1f0
//
__int64 __fastcall sub_171D1F0(__int64 a1, __int64 a2, __int64 a3, __int64 **a4, double a5, double a6, double a7)
{
  unsigned __int8 v9; // al
  unsigned __int8 v10; // al
  __int64 v11; // r14
  __int64 v12; // r15
  __int64 v13; // rdi
  unsigned __int8 v14; // al
  char v15; // r13
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned int v21; // edx
  bool v22; // si
  unsigned int v23; // esi
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  int v27; // edx
  bool v28; // di
  int v29; // eax
  unsigned int v30; // edx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // rax
  __int64 v35; // r10
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v42; // rdi
  unsigned __int8 v43; // al
  __int64 v44; // r14
  __int64 *v45; // rax
  __int64 v46; // rax
  unsigned __int8 v47; // al
  __int64 *v48; // rax
  __int64 v49; // rax
  __int64 v50; // r13
  bool v51; // cc
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 *v54; // r12
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 *v57; // rsi
  __int64 v58; // rdi
  __int64 v59; // rdx
  bool v60; // zf
  __int64 v61; // rsi
  __int64 v62; // rsi
  unsigned __int8 *v63; // rsi
  __int64 v64; // rax
  __int64 *v65; // r15
  __int64 v66; // rax
  __int64 v67; // rcx
  __int64 v68; // rsi
  __int64 v69; // rsi
  unsigned __int8 *v70; // rsi
  __int64 v71; // [rsp+8h] [rbp-D8h]
  __int64 v72; // [rsp+10h] [rbp-D0h]
  __int64 v73; // [rsp+18h] [rbp-C8h]
  __int64 v74; // [rsp+18h] [rbp-C8h]
  __int64 v75; // [rsp+18h] [rbp-C8h]
  __int64 v76; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v77; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v78[2]; // [rsp+30h] [rbp-B0h] BYREF
  char v79; // [rsp+40h] [rbp-A0h]
  char v80; // [rsp+41h] [rbp-9Fh]
  __int64 v81[2]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v82; // [rsp+60h] [rbp-80h]
  _QWORD v83[2]; // [rsp+70h] [rbp-70h] BYREF
  __int16 v84; // [rsp+80h] [rbp-60h]
  __int64 v85[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v86; // [rsp+A0h] [rbp-40h]

  v9 = *(_BYTE *)(a2 + 16);
  if ( v9 <= 0x17u )
  {
    if ( v9 != 5 || *(_WORD *)(a2 + 18) != 32 )
      goto LABEL_3;
LABEL_50:
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    {
      v42 = **(_QWORD **)(a2 - 8);
      if ( a3 != v42 )
      {
LABEL_52:
        v43 = *(_BYTE *)(a3 + 16);
        if ( v43 <= 0x17u )
        {
          if ( v43 != 5 || *(_WORD *)(a3 + 18) != 32 )
            return 0;
        }
        else if ( v43 != 56 )
        {
          return 0;
        }
        v74 = a3;
        v44 = sub_1649C60(v42);
        if ( (*(_BYTE *)(v74 + 23) & 0x40) != 0 )
          v45 = *(__int64 **)(v74 - 8);
        else
          v45 = (__int64 *)(v74 - 24LL * (*(_DWORD *)(v74 + 20) & 0xFFFFFFF));
        v46 = sub_1649C60(*v45);
        a3 = v74;
        if ( v44 != v46 )
          goto LABEL_3;
        v11 = v74;
LABEL_60:
        v47 = *(_BYTE *)(a3 + 16);
        if ( v47 <= 0x17u )
        {
          if ( v47 == 5 && *(_WORD *)(a3 + 18) == 32 )
            goto LABEL_62;
        }
        else if ( v47 == 56 )
        {
LABEL_62:
          v12 = a2;
          goto LABEL_6;
        }
        v12 = a2;
        v15 = 0;
        goto LABEL_13;
      }
    }
    else
    {
      v42 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      if ( a3 != v42 )
        goto LABEL_52;
    }
    v11 = 0;
    goto LABEL_60;
  }
  if ( v9 == 56 )
    goto LABEL_50;
LABEL_3:
  v10 = *(_BYTE *)(a3 + 16);
  if ( v10 <= 0x17u )
  {
    if ( v10 != 5 || *(_WORD *)(a3 + 18) != 32 )
      return 0;
  }
  else if ( v10 != 56 )
  {
    return 0;
  }
  v11 = 0;
  v12 = 0;
LABEL_6:
  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
  {
    v13 = **(_QWORD **)(a3 - 8);
    if ( a2 != v13 )
      goto LABEL_8;
LABEL_64:
    v12 = a3;
    v15 = 1;
    goto LABEL_13;
  }
  v13 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
  if ( a2 == v13 )
    goto LABEL_64;
LABEL_8:
  v14 = *(_BYTE *)(a2 + 16);
  if ( v14 > 0x17u )
  {
    if ( v14 != 56 )
      goto LABEL_11;
  }
  else if ( v14 != 5 || *(_WORD *)(a2 + 18) != 32 )
  {
    goto LABEL_11;
  }
  v71 = a3;
  v72 = a3;
  v75 = sub_1649C60(v13);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v48 = *(__int64 **)(a2 - 8);
  else
    v48 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v49 = sub_1649C60(*v48);
  v16 = a2;
  v17 = v72;
  if ( v75 == v49 )
  {
    v11 = a2;
    v12 = v71;
    v15 = 1;
    goto LABEL_15;
  }
LABEL_11:
  if ( !v12 )
    return 0;
  v15 = 0;
LABEL_13:
  if ( !v11 )
  {
    v36 = sub_170B0F0(a1, v12, a5, a6, a7);
    goto LABEL_38;
  }
  v16 = v11;
  v17 = v12;
LABEL_15:
  v18 = 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
  v19 = v17 - v18;
  if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
  {
    v19 = *(_QWORD *)(v12 - 8);
    v17 = v19 + v18;
  }
  v20 = v19 + 24;
  if ( v20 == v17 )
  {
    v23 = 0;
  }
  else
  {
    v21 = 0;
    do
    {
      v22 = *(_BYTE *)(*(_QWORD *)v20 + 16LL) != 13;
      v20 += 24;
      v21 += v22;
    }
    while ( v20 != v17 );
    v23 = v21;
  }
  v24 = 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
  v25 = v16 - v24;
  if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
  {
    v25 = *(_QWORD *)(v11 - 8);
    v16 = v25 + v24;
  }
  v26 = v25 + 24;
  if ( v26 == v16 )
  {
    v30 = v23;
    v29 = 0;
  }
  else
  {
    v27 = 0;
    do
    {
      v28 = *(_BYTE *)(*(_QWORD *)v26 + 16LL) != 13;
      v26 += 24;
      v27 += v28;
    }
    while ( v26 != v16 );
    v29 = v27;
    v30 = v23 + v27;
  }
  if ( v30 > 1 )
  {
    if ( v23 )
    {
      v31 = *(_QWORD *)(v12 + 8);
      if ( !v31 || *(_QWORD *)(v31 + 8) )
        return 0;
    }
    if ( v29 )
    {
      v32 = *(_QWORD *)(v11 + 8);
      if ( !v32 || *(_QWORD *)(v32 + 8) )
        return 0;
    }
  }
  v33 = sub_170B0F0(a1, v12, a5, a6, a7);
  v34 = sub_170B0F0(a1, v11, a5, a6, a7);
  v35 = *(_QWORD *)(a1 + 8);
  v86 = 257;
  if ( *(_BYTE *)(v33 + 16) > 0x10u || *(_BYTE *)(v34 + 16) > 0x10u )
  {
    v36 = (__int64)sub_170A2B0(v35, 13, (__int64 *)v33, v34, v85, 0, 0);
  }
  else
  {
    v73 = v35;
    v36 = sub_15A2B60((__int64 *)v33, v34, 0, 0, a5, a6, a7);
    v37 = sub_14DBA30(v36, *(_QWORD *)(v73 + 96), 0);
    if ( v37 )
      v36 = v37;
  }
LABEL_38:
  if ( v15 )
  {
    v80 = 1;
    v50 = *(_QWORD *)(a1 + 8);
    v79 = 3;
    v51 = *(_BYTE *)(v36 + 16) <= 0x10u;
    v78[0] = (__int64)"diff.neg";
    if ( v51 )
    {
      v36 = sub_15A2B90((__int64 *)v36, 0, 0, v38, a5, a6, a7);
      v52 = sub_14DBA30(v36, *(_QWORD *)(v50 + 96), 0);
      if ( v52 )
        v36 = v52;
    }
    else
    {
      v84 = 257;
      v36 = sub_15FB530((__int64 *)v36, (__int64)v83, 0, v38);
      v64 = *(_QWORD *)(v50 + 8);
      if ( v64 )
      {
        v65 = *(__int64 **)(v50 + 16);
        sub_157E9D0(v64 + 40, v36);
        v66 = *(_QWORD *)(v36 + 24);
        v67 = *v65;
        *(_QWORD *)(v36 + 32) = v65;
        v67 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v36 + 24) = v67 | v66 & 7;
        *(_QWORD *)(v67 + 8) = v36 + 24;
        *v65 = *v65 & 7 | (v36 + 24);
      }
      v57 = v78;
      v58 = v36;
      sub_164B780(v36, v78);
      v60 = *(_QWORD *)(v50 + 80) == 0;
      v76 = v36;
      if ( v60 )
        goto LABEL_102;
      (*(void (__fastcall **)(__int64, __int64 *))(v50 + 88))(v50 + 64, &v76);
      v68 = *(_QWORD *)v50;
      if ( *(_QWORD *)v50 )
      {
        v85[0] = *(_QWORD *)v50;
        sub_1623A60((__int64)v85, v68, 2);
        v69 = *(_QWORD *)(v36 + 48);
        if ( v69 )
          sub_161E7C0(v36 + 48, v69);
        v70 = (unsigned __int8 *)v85[0];
        *(_QWORD *)(v36 + 48) = v85[0];
        if ( v70 )
          sub_1623210((__int64)v85, v70, v36 + 48);
      }
    }
  }
  v39 = *(_QWORD *)(a1 + 8);
  v82 = 257;
  if ( a4 == *(__int64 ***)v36 )
    return v36;
  if ( *(_BYTE *)(v36 + 16) <= 0x10u )
  {
    v36 = sub_15A4750((__int64 ***)v36, a4, 1);
    v40 = sub_14DBA30(v36, *(_QWORD *)(v39 + 96), 0);
    if ( v40 )
      return v40;
    return v36;
  }
  v86 = 257;
  v36 = sub_15FE0A0((_QWORD *)v36, (__int64)a4, 1, (__int64)v85, 0);
  v53 = *(_QWORD *)(v39 + 8);
  if ( v53 )
  {
    v54 = *(__int64 **)(v39 + 16);
    sub_157E9D0(v53 + 40, v36);
    v55 = *(_QWORD *)(v36 + 24);
    v56 = *v54;
    *(_QWORD *)(v36 + 32) = v54;
    v56 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v36 + 24) = v56 | v55 & 7;
    *(_QWORD *)(v56 + 8) = v36 + 24;
    *v54 = *v54 & 7 | (v36 + 24);
  }
  v57 = v81;
  v58 = v36;
  sub_164B780(v36, v81);
  v60 = *(_QWORD *)(v39 + 80) == 0;
  v77 = v36;
  if ( v60 )
LABEL_102:
    sub_4263D6(v58, v57, v59);
  (*(void (__fastcall **)(__int64, __int64 *))(v39 + 88))(v39 + 64, &v77);
  v61 = *(_QWORD *)v39;
  if ( *(_QWORD *)v39 )
  {
    v83[0] = *(_QWORD *)v39;
    sub_1623A60((__int64)v83, v61, 2);
    v62 = *(_QWORD *)(v36 + 48);
    if ( v62 )
      sub_161E7C0(v36 + 48, v62);
    v63 = (unsigned __int8 *)v83[0];
    *(_QWORD *)(v36 + 48) = v83[0];
    if ( v63 )
      sub_1623210((__int64)v83, v63, v36 + 48);
  }
  return v36;
}
