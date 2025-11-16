// Function: sub_11860A0
// Address: 0x11860a0
//
_QWORD *__fastcall sub_11860A0(__int64 a1, __int64 a2)
{
  unsigned int v4; // eax
  unsigned int v5; // r15d
  _QWORD *v6; // r10
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int8 v11; // si
  __int64 v12; // r13
  unsigned __int8 v13; // dl
  int v14; // r12d
  __int64 v15; // rax
  __int64 v16; // rax
  _BYTE *v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rcx
  unsigned __int8 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int8 *v25; // rax
  __int64 v26; // r12
  __int64 v27; // rax
  unsigned __int64 v28; // rax
  int v29; // edx
  unsigned int v30; // r12d
  __int64 *v31; // rax
  __int64 v32; // rax
  __int64 v33; // r10
  __int64 v34; // r12
  __int64 v35; // r14
  __int64 v36; // r13
  bool v37; // al
  _QWORD *v38; // rax
  __int64 v39; // r13
  __int64 v40; // rbx
  __int64 v41; // rdx
  unsigned int v42; // esi
  char v43; // [rsp+Fh] [rbp-B1h]
  char v44; // [rsp+18h] [rbp-A8h]
  _BYTE *v45; // [rsp+18h] [rbp-A8h]
  __int64 v46; // [rsp+18h] [rbp-A8h]
  __int64 v47; // [rsp+18h] [rbp-A8h]
  __int64 v48; // [rsp+18h] [rbp-A8h]
  __int64 v49; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *v50; // [rsp+20h] [rbp-A0h]
  unsigned __int8 *v51; // [rsp+28h] [rbp-98h]
  __int64 v52; // [rsp+28h] [rbp-98h]
  __int64 v53; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int8 *v54; // [rsp+38h] [rbp-88h]
  __int64 v55; // [rsp+40h] [rbp-80h]
  __int16 v56; // [rsp+50h] [rbp-70h]
  __int64 v57; // [rsp+60h] [rbp-60h] BYREF
  __int64 v58; // [rsp+68h] [rbp-58h]
  __int64 *v59; // [rsp+70h] [rbp-50h] BYREF
  __int16 v60; // [rsp+80h] [rbp-40h]

  v4 = sub_BCB060(*(_QWORD *)(a1 + 8));
  if ( !v4 )
    return 0;
  v5 = v4;
  if ( (v4 & (v4 - 1)) != 0 )
    return 0;
  v8 = *(_QWORD *)(a1 - 32);
  v9 = *(_QWORD *)(v8 + 16);
  if ( !v9 )
    return 0;
  v6 = *(_QWORD **)(v9 + 8);
  if ( v6 )
    return 0;
  if ( *(_BYTE *)v8 != 58 )
    return 0;
  v10 = *(_QWORD *)(v8 - 64);
  v11 = *(_BYTE *)v10;
  if ( *(_BYTE *)v10 <= 0x1Cu )
    return 0;
  if ( (unsigned int)v11 - 42 > 0x11 )
    return 0;
  v12 = *(_QWORD *)(v8 - 32);
  v13 = *(_BYTE *)v12;
  if ( *(_BYTE *)v12 <= 0x1Cu )
    return 0;
  v14 = v13;
  if ( (unsigned int)v13 - 42 > 0x11 )
    return 0;
  v15 = *(_QWORD *)(v10 + 16);
  if ( !v15 )
    return v6;
  if ( *(_QWORD *)(v15 + 8) )
    return v6;
  v44 = *(_BYTE *)v12;
  if ( (unsigned int)v11 - 54 > 1 )
    return v6;
  v16 = sub_986520(v10);
  v6 = 0;
  v50 = *(unsigned __int8 **)v16;
  if ( !*(_QWORD *)v16 )
    return v6;
  v17 = *(_BYTE **)(v16 + 32);
  if ( *v17 == 68 && *((_QWORD *)v17 - 4) )
    v17 = (_BYTE *)*((_QWORD *)v17 - 4);
  v18 = *(_QWORD *)(v12 + 16);
  if ( !v18 )
    return v6;
  if ( *(_QWORD *)(v18 + 8) )
    return v6;
  v43 = v44;
  v45 = v17;
  if ( (unsigned int)(v14 - 54) > 1 )
    return v6;
  v19 = sub_986520(v12);
  v6 = 0;
  v51 = *(unsigned __int8 **)v19;
  if ( !*(_QWORD *)v19 )
    return v6;
  v20 = *(_QWORD *)(v19 + 32);
  v21 = (__int64)v45;
  if ( *(_BYTE *)v20 == 68 && *(_QWORD *)(v20 - 32) )
    v20 = *(_QWORD *)(v20 - 32);
  if ( v11 == v43 )
    return v6;
  if ( v11 == 55 )
  {
    v21 = v20;
    v20 = (__int64)v45;
    v22 = v50;
    v50 = v51;
    v51 = v22;
  }
  v58 = v21;
  v57 = v5;
  v23 = *(_QWORD *)(v20 + 16);
  if ( v23
    && !*(_QWORD *)(v23 + 8)
    && *(_BYTE *)v20 == 44
    && (v49 = v21, v37 = sub_F17ED0(&v57, *(_QWORD *)(v20 - 64)), v21 = v49, v37)
    && *(_QWORD *)(v20 - 32) == v58 )
  {
    v25 = *(unsigned __int8 **)(a1 - 64);
    v20 = v49;
  }
  else
  {
    v57 = v5;
    v58 = v20;
    v24 = *(_QWORD *)(v21 + 16);
    if ( !v24 )
      return 0;
    if ( *(_QWORD *)(v24 + 8) )
      return 0;
    if ( *(_BYTE *)v21 != 44 )
      return 0;
    v46 = v21;
    if ( !sub_F17ED0(&v57, *(_QWORD *)(v21 - 64)) )
      return 0;
    v21 = v46;
    if ( *(_QWORD *)(v46 - 32) != v58 )
      return 0;
    v25 = *(unsigned __int8 **)(a1 - 64);
    if ( v46 != v20 )
    {
      if ( v25 != v51 )
        return 0;
      goto LABEL_33;
    }
  }
  if ( v50 != v25 )
    return 0;
LABEL_33:
  v26 = *(_QWORD *)(a1 - 96);
  v57 = 32;
  v58 = v20;
  v59 = 0;
  v27 = *(_QWORD *)(v26 + 16);
  if ( !v27 )
    return 0;
  if ( *(_QWORD *)(v27 + 8) )
    return 0;
  if ( *(_BYTE *)v26 != 82 )
    return 0;
  v47 = v21;
  v28 = sub_B53900(v26);
  v53 = sub_B53630(v28, v57);
  LODWORD(v54) = v29;
  if ( !(_BYTE)v29 || *(_QWORD *)(v26 - 64) != v58 || !(unsigned __int8)sub_10081F0(&v59, *(_QWORD *)(v26 - 32)) )
    return 0;
  if ( v50 == v51 )
  {
    v30 = (v47 != v20) + 180;
  }
  else if ( v47 == v20 )
  {
    v30 = 180;
    if ( !sub_98ED70(v51, 0, 0, 0, 0) )
    {
      v60 = 257;
      v51 = (unsigned __int8 *)sub_1156690((__int64 *)a2, (__int64)v51, (__int64)&v57);
    }
  }
  else
  {
    v30 = 181;
    if ( !sub_98ED70(v50, 0, 0, 0, 0) )
    {
      v60 = 257;
      v50 = (unsigned __int8 *)sub_1156690((__int64 *)a2, (__int64)v50, (__int64)&v57);
    }
  }
  v57 = *(_QWORD *)(a1 + 8);
  v31 = (__int64 *)sub_B43CA0(a1);
  v32 = sub_B6E160(v31, v30, (__int64)&v57, 1);
  v33 = *(_QWORD *)(a1 + 8);
  v56 = 257;
  v34 = v32;
  if ( v33 == *(_QWORD *)(v20 + 8) )
  {
    v35 = v20;
  }
  else
  {
    v48 = v33;
    v35 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a2 + 80) + 120LL))(
            *(_QWORD *)(a2 + 80),
            39,
            v20,
            v33);
    if ( !v35 )
    {
      v60 = 257;
      v38 = sub_BD2C40(72, unk_3F10A14);
      v35 = (__int64)v38;
      if ( v38 )
        sub_B515B0((__int64)v38, v20, v48, (__int64)&v57, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v35,
        &v53,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v39 = *(_QWORD *)a2;
      v40 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      while ( v40 != v39 )
      {
        v41 = *(_QWORD *)(v39 + 8);
        v42 = *(_DWORD *)v39;
        v39 += 16;
        sub_B99FD0(v35, v42, v41);
      }
    }
  }
  v55 = v35;
  v36 = 0;
  v60 = 257;
  v53 = (__int64)v50;
  v54 = v51;
  if ( v34 )
    v36 = *(_QWORD *)(v34 + 24);
  v6 = sub_BD2CC0(88, 4u);
  if ( v6 )
  {
    v52 = (__int64)v6;
    sub_B44260((__int64)v6, **(_QWORD **)(v36 + 16), 56, 4u, 0, 0);
    *(_QWORD *)(v52 + 72) = 0;
    sub_B4A290(v52, v36, v34, &v53, 3, (__int64)&v57, 0, 0);
    return (_QWORD *)v52;
  }
  return v6;
}
