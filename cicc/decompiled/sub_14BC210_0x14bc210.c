// Function: sub_14BC210
// Address: 0x14bc210
//
__int64 __fastcall sub_14BC210(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  unsigned int v8; // edx
  unsigned int v9; // r10d
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  int v13; // eax
  __int64 v14; // r12
  char v15; // al
  unsigned int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rcx
  int v21; // eax
  __int64 v22; // rax
  unsigned int v23; // eax
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  __int64 *v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rax
  char v29; // al
  __int64 v30; // rbx
  unsigned __int64 v31; // rax
  __int64 v32; // rcx
  int v33; // edx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // rax
  _BYTE *v37; // rdi
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  void *v40; // rcx
  unsigned __int64 v41; // rdx
  void *v42; // rcx
  unsigned __int64 v43; // rdx
  void *v44; // rcx
  unsigned __int8 v45; // dl
  char v46; // dl
  _BYTE *v47; // rdi
  unsigned __int8 v48; // r10
  char v49; // al
  char v50; // al
  unsigned __int64 v51; // rax
  void *v52; // rcx
  char v53; // al
  __int64 v54; // rax
  unsigned __int8 v55; // [rsp+Fh] [rbp-81h]
  unsigned __int8 v56; // [rsp+Fh] [rbp-81h]
  unsigned __int8 v57; // [rsp+Fh] [rbp-81h]
  unsigned __int8 v58; // [rsp+Fh] [rbp-81h]
  unsigned __int8 v59; // [rsp+Fh] [rbp-81h]
  unsigned __int8 v60; // [rsp+Fh] [rbp-81h]
  unsigned __int8 v61; // [rsp+Fh] [rbp-81h]
  unsigned __int8 v62; // [rsp+Fh] [rbp-81h]
  unsigned __int8 v63; // [rsp+Fh] [rbp-81h]
  char v64; // [rsp+10h] [rbp-80h] BYREF
  __int64 *v65; // [rsp+18h] [rbp-78h] BYREF
  _BYTE *v66; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v67; // [rsp+28h] [rbp-68h] BYREF
  __int64 v68; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v69; // [rsp+38h] [rbp-58h] BYREF
  unsigned __int64 v70; // [rsp+40h] [rbp-50h] BYREF
  __int64 *v71; // [rsp+48h] [rbp-48h] BYREF
  __int64 v72[8]; // [rsp+50h] [rbp-40h] BYREF

  LOBYTE(v8) = sub_15FF820(a1) & (a2 == a3);
  v9 = v8;
  if ( (_BYTE)v8 )
    return v9;
  if ( (_DWORD)a1 == 37 )
  {
    v70 = a2;
    v71 = (__int64 *)&v64;
    v19 = *(unsigned __int8 *)(a3 + 16);
    if ( (unsigned __int8)v19 <= 0x17u )
    {
      if ( (_BYTE)v19 != 5 )
        goto LABEL_28;
      v43 = *(unsigned __int16 *)(a3 + 18);
      if ( (unsigned __int16)v43 > 0x17u )
        goto LABEL_28;
      v44 = &loc_80A800;
      v21 = (unsigned __int16)v43;
      if ( !_bittest64((const __int64 *)&v44, v43) )
        goto LABEL_28;
    }
    else
    {
      if ( (unsigned __int8)v19 > 0x2Fu )
        goto LABEL_28;
      v20 = 0x80A800000000LL;
      if ( !_bittest64(&v20, v19) )
        goto LABEL_28;
      v21 = (unsigned __int8)v19 - 24;
    }
    if ( v21 == 11 && (*(_BYTE *)(a3 + 17) & 2) != 0 )
    {
      v56 = v9;
      v22 = sub_13CF970(a3);
      v9 = v56;
      if ( a2 == *(_QWORD *)v22 )
      {
        v23 = sub_13D2630(&v71, *(_BYTE **)(v22 + 24));
        v9 = v56;
        if ( (_BYTE)v23 )
          return v23;
      }
    }
LABEL_28:
    v70 = (unsigned __int64)&v65;
    v71 = (__int64 *)&v66;
    v24 = *(unsigned __int8 *)(a2 + 16);
    if ( (unsigned __int8)v24 <= 0x17u )
    {
      if ( (_BYTE)v24 != 5 )
        return v9;
      v41 = *(unsigned __int16 *)(a2 + 18);
      if ( (unsigned __int16)v41 > 0x17u )
        goto LABEL_46;
      v42 = &loc_80A800;
      if ( !_bittest64((const __int64 *)&v42, v41) || (_WORD)v41 != 11 )
        goto LABEL_46;
    }
    else
    {
      if ( (unsigned __int8)v24 > 0x2Fu )
      {
        if ( (_BYTE)v24 != 51 )
          return v9;
        goto LABEL_66;
      }
      v25 = 0x80A800000000LL;
      if ( !_bittest64(&v25, v24) || (_BYTE)v24 != 35 )
        return v9;
    }
    if ( (*(_BYTE *)(a2 + 17) & 2) != 0 )
    {
      v26 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
          ? *(__int64 **)(a2 - 8)
          : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v27 = *v26;
      if ( v27 )
      {
        v57 = v9;
        v65 = (__int64 *)v27;
        v28 = sub_13CF970(a2);
        v29 = sub_13D2630(&v71, *(_BYTE **)(v28 + 24));
        v9 = v57;
        if ( !v29 )
          goto LABEL_43;
        v30 = (__int64)v65;
        v69 = &v67;
        v31 = *(unsigned __int8 *)(a3 + 16);
        v68 = (__int64)v65;
        if ( (unsigned __int8)v31 <= 0x17u )
        {
          if ( (_BYTE)v31 != 5 )
            goto LABEL_43;
          v51 = *(unsigned __int16 *)(a3 + 18);
          if ( (unsigned __int16)v51 > 0x17u )
            goto LABEL_43;
          v52 = &loc_80A800;
          v33 = (unsigned __int16)v51;
          if ( !_bittest64((const __int64 *)&v52, v51) )
            goto LABEL_43;
        }
        else
        {
          if ( (unsigned __int8)v31 > 0x2Fu )
            goto LABEL_43;
          v32 = 0x80A800000000LL;
          v33 = (unsigned __int8)v31 - 24;
          if ( !_bittest64(&v32, v31) )
            goto LABEL_43;
        }
        if ( v33 != 11 || (*(_BYTE *)(a3 + 17) & 2) == 0 || (v34 = sub_13CF970(a3), v9 = v57, v30 != *(_QWORD *)v34) )
        {
LABEL_43:
          LOBYTE(v24) = *(_BYTE *)(a2 + 16);
          goto LABEL_44;
        }
        if ( (unsigned __int8)sub_13D2630(&v69, *(_BYTE **)(v34 + 24)) )
          goto LABEL_97;
        LOBYTE(v24) = *(_BYTE *)(a2 + 16);
        v9 = v57;
      }
    }
LABEL_44:
    if ( (_BYTE)v24 != 51 )
    {
      if ( (_BYTE)v24 != 5 )
        return v9;
LABEL_46:
      if ( *(_WORD *)(a2 + 18) != 27 )
        return v9;
      v35 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v36 = *(__int64 **)(a2 - 24 * v35);
      if ( !v36 )
        return v9;
      v65 = *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v37 = *(_BYTE **)(a2 + 24 * (1 - v35));
      if ( v37[16] != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v37 + 8LL) != 16 )
          return v9;
LABEL_50:
        v58 = v9;
        v38 = sub_15A1020(v37);
        v9 = v58;
        if ( !v38 || *(_BYTE *)(v38 + 16) != 13 )
          return v9;
        v66 = (_BYTE *)(v38 + 24);
        v36 = v65;
        goto LABEL_69;
      }
      goto LABEL_68;
    }
LABEL_66:
    v36 = *(__int64 **)(a2 - 48);
    if ( !v36 )
      return v9;
    v37 = *(_BYTE **)(a2 - 24);
    v65 = *(__int64 **)(a2 - 48);
    v45 = v37[16];
    if ( v45 != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v37 + 8LL) != 16 || v45 > 0x10u )
        return v9;
      goto LABEL_50;
    }
LABEL_68:
    v66 = v37 + 24;
LABEL_69:
    v70 = (unsigned __int64)v36;
    v71 = (__int64 *)&v67;
    v46 = *(_BYTE *)(a3 + 16);
    if ( v46 == 51 )
    {
      if ( *(__int64 **)(a3 - 48) != v36 )
        return v9;
      v62 = v9;
      v53 = sub_13D2630(&v71, *(_BYTE **)(a3 - 24));
      v9 = v62;
      if ( !v53 )
        return v9;
    }
    else
    {
      if ( v46 != 5 || *(_WORD *)(a3 + 18) != 27 || *(__int64 **)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)) != v36 )
        return v9;
      v47 = *(_BYTE **)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
      if ( v47[16] == 13 )
      {
        v67 = v47 + 24;
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v47 + 8LL) != 16 )
          return v9;
        v63 = v9;
        v54 = sub_15A1020(v47);
        v9 = v63;
        if ( !v54 || *(_BYTE *)(v54 + 16) != 13 )
          return v9;
        *v71 = v54 + 24;
      }
    }
    v59 = v9;
    sub_14AA4E0((__int64)&v70, *((_DWORD *)v66 + 2));
    sub_14BB090((__int64)v65, (__int64)&v70, a4, a5 + 1, 0, 0, 0, 0);
    v48 = v59;
    if ( *((_DWORD *)v66 + 2) <= 0x40u )
    {
      if ( (*(_QWORD *)v66 & ~v70) != 0 )
        goto LABEL_79;
    }
    else
    {
      v49 = sub_16A5A00(v66, &v70);
      v48 = v59;
      if ( !v49 )
      {
LABEL_79:
        v61 = v48;
        sub_135E100(v72);
        sub_135E100((__int64 *)&v70);
        return v61;
      }
    }
    if ( *((_DWORD *)v67 + 2) <= 0x40u )
    {
      if ( (*(_QWORD *)v67 & ~v70) != 0 )
        goto LABEL_79;
    }
    else
    {
      v60 = v48;
      v50 = sub_16A5A00(v67, &v70);
      v48 = v60;
      if ( !v50 )
        goto LABEL_79;
    }
    sub_135E100(v72);
    sub_135E100((__int64 *)&v70);
LABEL_97:
    LOBYTE(v9) = (int)sub_16A9900(v66, v67) <= 0;
    return v9;
  }
  if ( (_DWORD)a1 != 41 )
    return v9;
  v70 = a2;
  v71 = &v68;
  v11 = *(unsigned __int8 *)(a3 + 16);
  if ( (unsigned __int8)v11 <= 0x17u )
  {
    if ( (_BYTE)v11 != 5 )
      return v9;
    v39 = *(unsigned __int16 *)(a3 + 18);
    if ( (unsigned __int16)v39 > 0x17u )
      return v9;
    v40 = &loc_80A800;
    v13 = (unsigned __int16)v39;
    if ( !_bittest64((const __int64 *)&v40, v39) )
      return v9;
  }
  else
  {
    if ( (unsigned __int8)v11 > 0x2Fu )
      return v9;
    v12 = 0x80A800000000LL;
    if ( !_bittest64(&v12, v11) )
      return v9;
    v13 = (unsigned __int8)v11 - 24;
  }
  if ( v13 == 11 && (*(_BYTE *)(a3 + 17) & 4) != 0 )
  {
    v14 = (*(_BYTE *)(a3 + 23) & 0x40) != 0 ? *(_QWORD *)(a3 - 8) : a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
    if ( a2 == *(_QWORD *)v14 )
    {
      v55 = v9;
      v15 = sub_13D2630(&v71, *(_BYTE **)(v14 + 24));
      v9 = v55;
      if ( v15 )
      {
        v16 = *(_DWORD *)(v68 + 8);
        v17 = 1LL << ((unsigned __int8)v16 - 1);
        if ( v16 > 0x40 )
          v18 = *(_QWORD *)(*(_QWORD *)v68 + 8LL * ((v16 - 1) >> 6));
        else
          v18 = *(_QWORD *)v68;
        LOBYTE(v9) = (v18 & v17) == 0;
      }
    }
  }
  return v9;
}
