// Function: sub_13D7050
// Address: 0x13d7050
//
__int64 __fastcall sub_13D7050(__int64 a1, _BYTE *a2, _QWORD *a3)
{
  unsigned __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // edx
  __int64 *v8; // rdx
  __int64 v9; // rcx
  unsigned __int8 v10; // dl
  _BYTE *v11; // r13
  __int64 result; // rax
  unsigned __int64 v13; // rcx
  void *v14; // rdi
  __int64 v15; // rdi
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // rsi
  int v20; // edx
  __int64 *v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned int v24; // r12d
  int v25; // r14d
  unsigned __int64 v26; // r14
  unsigned int v27; // r15d
  _QWORD *v28; // r13
  __int64 v29; // rcx
  int v30; // eax
  __int64 *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  bool v34; // zf
  __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  __int64 v37; // rdi
  int v38; // edx
  __int64 *v39; // rcx
  char v40; // al
  __int64 v41; // rdx
  __int64 v42; // rsi
  unsigned __int64 v43; // rsi
  void *v44; // rdi
  unsigned __int64 v45; // rcx
  void *v46; // rsi
  unsigned __int64 v47; // rdx
  void *v48; // rcx
  __int64 v49; // [rsp+8h] [rbp-78h]
  __int64 v50; // [rsp+8h] [rbp-78h]
  __int64 v51; // [rsp+18h] [rbp-68h] BYREF
  __int64 v52; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v53; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v54; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v55; // [rsp+38h] [rbp-48h] BYREF
  __int64 *v56; // [rsp+40h] [rbp-40h]
  unsigned int v57; // [rsp+48h] [rbp-38h]

  v5 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned __int8)v5 <= 0x17u )
  {
    if ( (_BYTE)v5 != 5 )
      goto LABEL_9;
    v13 = *(unsigned __int16 *)(a1 + 18);
    if ( (unsigned __int16)v13 > 0x17u )
      goto LABEL_9;
    v14 = &loc_80A800;
    v7 = (unsigned __int16)v13;
    if ( !_bittest64((const __int64 *)&v14, v13) )
      goto LABEL_9;
  }
  else
  {
    if ( (unsigned __int8)v5 > 0x2Fu )
      goto LABEL_9;
    v6 = 0x80A800000000LL;
    v7 = (unsigned __int8)v5 - 24;
    if ( !_bittest64(&v6, v5) )
      goto LABEL_9;
  }
  if ( v7 == 23 && (*(_BYTE *)(a1 + 17) & 2) != 0 )
  {
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    {
      v8 = *(__int64 **)(a1 - 8);
      v9 = *v8;
      if ( !*v8 )
        goto LABEL_9;
    }
    else
    {
      v8 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v9 = *v8;
      if ( !*v8 )
        goto LABEL_9;
    }
    v51 = v9;
    if ( a2 == (_BYTE *)v8[3] )
      return v51;
  }
LABEL_9:
  v10 = a2[16];
  v11 = a2 + 24;
  if ( v10 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 )
      return 0;
    if ( v10 > 0x10u )
      return 0;
    v5 = sub_15A1020(a2);
    if ( !v5 || *(_BYTE *)(v5 + 16) != 13 )
      return 0;
    v11 = (_BYTE *)(v5 + 24);
    LOBYTE(v5) = *(_BYTE *)(a1 + 16);
  }
  v54 = &v51;
  v55 = &v53;
  v56 = &v52;
  if ( (_BYTE)v5 == 51 )
  {
    v15 = *(_QWORD *)(a1 - 48);
    v16 = *(unsigned __int8 *)(v15 + 16);
    if ( (unsigned __int8)v16 > 0x17u )
    {
      if ( (unsigned __int8)v16 > 0x2Fu )
        goto LABEL_26;
      v29 = 0x80A800000000LL;
      if ( !_bittest64(&v29, v16) )
        goto LABEL_26;
      v30 = (unsigned __int8)v16 - 24;
    }
    else
    {
      if ( (_BYTE)v16 != 5 )
        goto LABEL_26;
      v47 = *(unsigned __int16 *)(v15 + 18);
      if ( (unsigned __int16)v47 > 0x17u )
        goto LABEL_26;
      v48 = &loc_80A800;
      v30 = (unsigned __int16)v47;
      if ( !_bittest64((const __int64 *)&v48, v47) )
        goto LABEL_26;
    }
    if ( v30 == 23 && (*(_BYTE *)(v15 + 17) & 2) != 0 )
    {
      v31 = (*(_BYTE *)(v15 + 23) & 0x40) != 0
          ? *(__int64 **)(v15 - 8)
          : (__int64 *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
      v32 = *v31;
      if ( v32 )
      {
        v51 = v32;
        v33 = sub_13CF970(v15);
        v34 = (unsigned __int8)sub_13D2630(&v55, *(_BYTE **)(v33 + 24)) == 0;
        v17 = *(_QWORD *)(a1 - 24);
        if ( !v34 && v17 )
        {
LABEL_40:
          *v56 = v17;
          goto LABEL_41;
        }
LABEL_27:
        v18 = *(unsigned __int8 *)(v17 + 16);
        if ( (unsigned __int8)v18 <= 0x17u )
        {
          if ( (_BYTE)v18 != 5 )
            return 0;
          v45 = *(unsigned __int16 *)(v17 + 18);
          if ( (unsigned __int16)v45 > 0x17u )
            return 0;
          v46 = &loc_80A800;
          v20 = (unsigned __int16)v45;
          if ( !_bittest64((const __int64 *)&v46, v45) )
            return 0;
        }
        else
        {
          if ( (unsigned __int8)v18 > 0x2Fu )
            return 0;
          v19 = 0x80A800000000LL;
          if ( !_bittest64(&v19, v18) )
            return 0;
          v20 = (unsigned __int8)v18 - 24;
        }
        if ( v20 != 23 || (*(_BYTE *)(v17 + 17) & 2) == 0 )
          return 0;
        v21 = (*(_BYTE *)(v17 + 23) & 0x40) != 0
            ? *(__int64 **)(v17 - 8)
            : (__int64 *)(v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF));
        v22 = *v21;
        if ( !v22 )
          return 0;
        *v54 = v22;
        v23 = (*(_BYTE *)(v17 + 23) & 0x40) != 0
            ? *(_QWORD *)(v17 - 8)
            : v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
        if ( !(unsigned __int8)sub_13D2630(&v55, *(_BYTE **)(v23 + 24)) )
          return 0;
        v17 = *(_QWORD *)(a1 - 48);
        if ( !v17 )
          return 0;
        goto LABEL_40;
      }
    }
LABEL_26:
    v17 = *(_QWORD *)(a1 - 24);
    goto LABEL_27;
  }
  if ( (_BYTE)v5 != 5 || *(_WORD *)(a1 + 18) != 27 )
    return 0;
  v35 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v36 = *(unsigned __int8 *)(v35 + 16);
  if ( (unsigned __int8)v36 <= 0x17u )
  {
    if ( (_BYTE)v36 != 5 )
      goto LABEL_97;
    v43 = *(unsigned __int16 *)(v35 + 18);
    if ( (unsigned __int16)v43 > 0x17u )
      goto LABEL_97;
    v44 = &loc_80A800;
    v38 = (unsigned __int16)v43;
    if ( !_bittest64((const __int64 *)&v44, v43) )
      goto LABEL_97;
  }
  else
  {
    if ( (unsigned __int8)v36 > 0x2Fu )
      goto LABEL_97;
    v37 = 0x80A800000000LL;
    if ( !_bittest64(&v37, v36) )
      goto LABEL_97;
    v38 = (unsigned __int8)v36 - 24;
  }
  if ( v38 != 23 || (*(_BYTE *)(v35 + 17) & 2) == 0 )
  {
LABEL_97:
    v42 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
    goto LABEL_80;
  }
  if ( (*(_BYTE *)(v35 + 23) & 0x40) != 0 )
  {
    if ( **(_QWORD **)(v35 - 8) )
    {
      v51 = **(_QWORD **)(v35 - 8);
      v39 = *(__int64 **)(v35 - 8);
      goto LABEL_78;
    }
    goto LABEL_97;
  }
  v39 = (__int64 *)(v35 - 24LL * (*(_DWORD *)(v35 + 20) & 0xFFFFFFF));
  if ( !*v39 )
    goto LABEL_97;
  v51 = *v39;
LABEL_78:
  v40 = sub_13D2630(&v55, (_BYTE *)v39[3]);
  v41 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( !v40 )
  {
    v42 = *(_QWORD *)(a1 + 24 * (1 - v41));
LABEL_80:
    if ( !sub_13D6F20(&v54, v42) )
      return 0;
    v17 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( !v17 )
      return 0;
    goto LABEL_40;
  }
  v42 = *(_QWORD *)(a1 + 24 * (1 - v41));
  if ( !v42 )
    goto LABEL_80;
  *v56 = v42;
LABEL_41:
  if ( *((_DWORD *)v11 + 2) > 0x40u )
  {
    if ( (unsigned __int8)sub_16A5220(v11, v53) )
      goto LABEL_43;
    return 0;
  }
  if ( *(_QWORD *)v11 != *v53 )
    return 0;
LABEL_43:
  sub_14C2530((unsigned int)&v54, v52, *a3, 0, a3[3], a3[4], a3[2], 0);
  v24 = (unsigned int)v55;
  if ( (unsigned int)v55 > 0x40 )
  {
    v25 = sub_16A5810(&v54);
  }
  else
  {
    v25 = 64;
    if ( (_QWORD)v54 << (64 - (unsigned __int8)v55) != -1 )
    {
      _BitScanReverse64(&v26, ~((_QWORD)v54 << (64 - (unsigned __int8)v55)));
      v25 = v26 ^ 0x3F;
    }
  }
  v27 = *((_DWORD *)v11 + 2);
  if ( v27 > 0x40 )
  {
    if ( v27 - (unsigned int)sub_16A57B0(v11) > 0x40 )
      goto LABEL_85;
    v28 = **(_QWORD ***)v11;
  }
  else
  {
    v28 = *(_QWORD **)v11;
  }
  if ( (unsigned int)sub_16431D0(*(_QWORD *)a1) - v25 > (unsigned __int64)v28 )
  {
    if ( v57 > 0x40 && v56 )
    {
      j_j___libc_free_0_0(v56);
      v24 = (unsigned int)v55;
    }
    if ( v24 > 0x40 )
    {
      if ( v54 )
        j_j___libc_free_0_0(v54);
    }
    return 0;
  }
LABEL_85:
  result = v51;
  if ( v57 > 0x40 && v56 )
  {
    v49 = v51;
    j_j___libc_free_0_0(v56);
    v24 = (unsigned int)v55;
    result = v49;
  }
  if ( v24 > 0x40 && v54 )
  {
    v50 = result;
    j_j___libc_free_0_0(v54);
    return v50;
  }
  return result;
}
