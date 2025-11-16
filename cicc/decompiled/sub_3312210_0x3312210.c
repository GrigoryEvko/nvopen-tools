// Function: sub_3312210
// Address: 0x3312210
//
__int64 __fastcall sub_3312210(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 (*v10)(); // rax
  __int128 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 v16; // rax
  _QWORD *v17; // r9
  __int64 v18; // r13
  __int64 v19; // r12
  int v20; // eax
  __int64 v21; // rbx
  __int64 v23; // r9
  char v24; // al
  int v25; // ebx
  __int64 v26; // r14
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // r14
  bool v30; // zf
  unsigned __int64 v31; // rdx
  __int64 v32; // [rsp-8h] [rbp-250h]
  unsigned int v33; // [rsp+1Ch] [rbp-22Ch]
  __int64 v34; // [rsp+20h] [rbp-228h]
  __int64 v35; // [rsp+28h] [rbp-220h]
  __int64 v36; // [rsp+30h] [rbp-218h]
  unsigned __int8 v38; // [rsp+43h] [rbp-205h]
  unsigned int v39; // [rsp+44h] [rbp-204h]
  __int64 v40; // [rsp+50h] [rbp-1F8h]
  unsigned __int8 v41; // [rsp+60h] [rbp-1E8h] BYREF
  char v42; // [rsp+61h] [rbp-1E7h] BYREF
  char v43; // [rsp+62h] [rbp-1E6h] BYREF
  char v44; // [rsp+63h] [rbp-1E5h] BYREF
  int v45; // [rsp+64h] [rbp-1E4h] BYREF
  __int64 v46; // [rsp+68h] [rbp-1E0h] BYREF
  int v47; // [rsp+70h] [rbp-1D8h]
  __int64 v48; // [rsp+78h] [rbp-1D0h] BYREF
  __int64 v49; // [rsp+80h] [rbp-1C8h]
  __int128 v50; // [rsp+88h] [rbp-1C0h] BYREF
  __int64 v51; // [rsp+98h] [rbp-1B0h] BYREF
  int v52; // [rsp+A0h] [rbp-1A8h]
  _QWORD *v53; // [rsp+A8h] [rbp-1A0h] BYREF
  __int64 v54; // [rsp+B0h] [rbp-198h]
  _QWORD v55[8]; // [rsp+B8h] [rbp-190h] BYREF
  __int64 v56; // [rsp+F8h] [rbp-150h] BYREF
  _QWORD *v57; // [rsp+100h] [rbp-148h]
  __int64 v58; // [rsp+108h] [rbp-140h]
  __int64 *v59; // [rsp+110h] [rbp-138h]
  _QWORD v60[38]; // [rsp+118h] [rbp-130h] BYREF

  v2 = a2;
  v3 = a1[1];
  v4 = *a1;
  v41 = 1;
  v42 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  LODWORD(v49) = 0;
  *(_QWORD *)&v50 = 0;
  DWORD2(v50) = 0;
  v45 = 0;
  v36 = v4;
  if ( !(unsigned __int8)sub_325E1D0(a2, 3u, 4u, &v41, &v42, (__int64)&v46, v3) )
    return 0;
  v7 = *(_QWORD *)(v46 + 56);
  if ( v7 )
  {
    if ( !*(_QWORD *)(v7 + 32) )
      return 0;
  }
  v39 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64))sub_33CA560)(a2, 3, v5, v6, v32);
  v8 = *(_QWORD *)(v46 + 56);
  v40 = v46;
  if ( !v8 )
    return 0;
  while ( 1 )
  {
    v9 = *(_QWORD *)(v8 + 16);
    if ( v2 == v9 )
      goto LABEL_5;
    if ( (unsigned int)(*(_DWORD *)(v9 + 24) - 56) > 1 )
      goto LABEL_5;
    v10 = *(__int64 (**)())(*(_QWORD *)v3 + 1896LL);
    if ( v10 == sub_302E070 )
      goto LABEL_5;
    v38 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *, __int128 *, int *, __int64))v10)(
            v3,
            v2,
            v9,
            &v48,
            &v50,
            &v45,
            v36);
    if ( !v38 )
      goto LABEL_5;
    v11 = v50;
    if ( (unsigned __int8)sub_33CF170(v50, *((_QWORD *)&v50 + 1)) )
      goto LABEL_5;
    v16 = *(unsigned int *)(v48 + 24);
    if ( (unsigned int)v16 <= 0x27 )
    {
      v13 = 0x8000008200LL;
      if ( _bittest64(&v13, v16) )
        goto LABEL_5;
    }
    v56 = 0;
    v57 = v60;
    v58 = 32;
    LODWORD(v59) = 0;
    BYTE4(v59) = 1;
    v33 = sub_33CA560(v11, *((_QWORD *)&v11 + 1), v12, v13, v14, v15);
    v18 = *(_QWORD *)(v48 + 56);
    if ( !v18 )
      break;
    v34 = v8;
    v35 = v2;
    while ( 1 )
    {
      v19 = *(_QWORD *)(v18 + 16);
      if ( v19 == v40 )
        goto LABEL_36;
      v20 = *(_DWORD *)(v19 + 24);
      if ( v20 > 365 )
      {
        if ( v20 > 470 )
        {
          if ( v20 == 497 )
            goto LABEL_22;
        }
        else if ( v20 > 464 )
        {
          goto LABEL_22;
        }
        goto LABEL_35;
      }
      if ( v20 > 337 )
        goto LABEL_22;
      if ( v20 <= 294 )
        break;
      if ( (unsigned int)(v20 - 298) <= 1 )
        goto LABEL_22;
LABEL_35:
      if ( (*(_BYTE *)(v19 + 32) & 2) != 0 )
        goto LABEL_22;
LABEL_36:
      v18 = *(_QWORD *)(v18 + 32);
      if ( !v18 )
      {
        v8 = v34;
        v2 = v35;
        goto LABEL_38;
      }
    }
    if ( v20 <= 292 && (*(_BYTE *)(v19 + 32) & 2) == 0 )
      goto LABEL_24;
LABEL_22:
    v43 = 1;
    v44 = 0;
    v51 = 0;
    v52 = 0;
    if ( (unsigned __int8)sub_325E1D0(v19, 3u, 4u, &v43, &v44, (__int64)&v51, v3) )
    {
      v53 = v55;
      v55[0] = v19;
      v54 = 0x200000001LL;
      v24 = sub_3285B00(v35, (__int64)&v56, (__int64)&v53, v33, 0, (__int64)v55);
      v17 = v55;
      if ( v24 )
      {
        v8 = v34;
        v2 = v35;
        if ( v53 != v55 )
          _libc_free((unsigned __int64)v53);
        goto LABEL_30;
      }
      if ( v53 != v55 )
        _libc_free((unsigned __int64)v53);
    }
    v20 = *(_DWORD *)(v19 + 24);
LABEL_24:
    if ( (unsigned int)(v20 - 56) > 1 )
      goto LABEL_36;
    v21 = *(_QWORD *)(v19 + 56);
    if ( !v21 )
      goto LABEL_36;
    while ( !(unsigned __int8)sub_3264B60(v19, *(_QWORD *)(v21 + 16), v36, v3) )
    {
      v21 = *(_QWORD *)(v21 + 32);
      if ( !v21 )
        goto LABEL_36;
    }
    v8 = v34;
    v2 = v35;
LABEL_30:
    if ( BYTE4(v59) )
      goto LABEL_5;
LABEL_31:
    _libc_free((unsigned __int64)v57);
    v8 = *(_QWORD *)(v8 + 32);
    if ( !v8 )
      return 0;
LABEL_6:
    v40 = v46;
  }
LABEL_38:
  if ( !BYTE4(v59) )
    _libc_free((unsigned __int64)v57);
  LODWORD(v59) = 0;
  v57 = v60;
  v58 = 0x100000020LL;
  v60[0] = v46;
  v53 = v55;
  BYTE4(v59) = 1;
  v56 = 1;
  v55[0] = v2;
  v55[1] = v9;
  v54 = 0x800000002LL;
  if ( (unsigned __int8)sub_3285B00(v2, (__int64)&v56, (__int64)&v53, v39, 0, (__int64)v17)
    || (unsigned __int8)sub_3285B00(v9, (__int64)&v56, (__int64)&v53, v39, 0, v23) )
  {
    if ( v53 != v55 )
      _libc_free((unsigned __int64)v53);
    if ( !BYTE4(v59) )
      goto LABEL_31;
LABEL_5:
    v8 = *(_QWORD *)(v8 + 32);
    if ( !v8 )
      return 0;
    goto LABEL_6;
  }
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  if ( !BYTE4(v59) )
    _libc_free((unsigned __int64)v57);
  v25 = v45;
  v26 = *a1;
  v27 = *(_QWORD *)(v2 + 80);
  if ( v42 )
  {
    v56 = *(_QWORD *)(v2 + 80);
    if ( v41 )
    {
      if ( v27 )
        sub_B96E90((__int64)&v56, v27, 1);
      LODWORD(v57) = *(_DWORD *)(v2 + 72);
      v28 = sub_33E95C0(v26, v2, 0, (unsigned int)&v56, v48, v49, v50, *((__int64 *)&v50 + 1), v25);
    }
    else
    {
      if ( v27 )
        sub_B96E90((__int64)&v56, v27, 1);
      LODWORD(v57) = *(_DWORD *)(v2 + 72);
      v28 = sub_33F6C50(v26, v2, 0, (unsigned int)&v56, v48, v49, v50, *((__int64 *)&v50 + 1), v25);
    }
  }
  else
  {
    v56 = *(_QWORD *)(v2 + 80);
    if ( v41 )
    {
      if ( v27 )
        sub_B96E90((__int64)&v56, v27, 1);
      LODWORD(v57) = *(_DWORD *)(v2 + 72);
      v28 = sub_33EA400(v26, v2, 0, (unsigned int)&v56, v48, v49, v50, *((__int64 *)&v50 + 1), v25);
    }
    else
    {
      if ( v27 )
        sub_B96E90((__int64)&v56, v27, 1);
      LODWORD(v57) = *(_DWORD *)(v2 + 72);
      v28 = sub_33EA4E0(v26, v2, 0, (unsigned int)&v56, v48, v49, v50, v25);
    }
  }
  v29 = v28;
  if ( v56 )
    sub_B91220((__int64)&v56, v56);
  v30 = v41 == 0;
  v31 = *(_QWORD *)(*a1 + 768);
  v58 = *a1;
  v57 = (_QWORD *)v31;
  *(_QWORD *)(v58 + 768) = &v56;
  v56 = (__int64)off_4A360B8;
  v59 = a1;
  if ( v30 )
  {
    sub_34161C0(*a1, v2, 0, v29, 1);
  }
  else
  {
    sub_34161C0(*a1, v2, 0, v29, 0);
    sub_34161C0(*a1, v2, 1, v29, 2);
  }
  sub_32EB240((__int64)a1, v2);
  sub_34161C0(*a1, v9, 0, v29, v41);
  sub_32EB240((__int64)a1, v9);
  *(_QWORD *)(v58 + 768) = v57;
  return v38;
}
