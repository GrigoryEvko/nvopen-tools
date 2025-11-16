// Function: sub_29CEC00
// Address: 0x29cec00
//
__int64 __fastcall sub_29CEC00(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v3; // rdi
  __int64 v4; // rcx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rbx
  char v7; // al
  bool v8; // zf
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 *v18; // r13
  const char *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // r14
  const char *v23; // rax
  _QWORD *v24; // rdi
  _QWORD *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r13
  __int64 v28; // rax
  int v29; // eax
  __int64 v30; // rax
  __int64 v31; // r14
  unsigned __int16 v32; // r12
  _QWORD *v33; // rax
  __int64 v34; // r13
  __int64 v35; // r12
  unsigned __int8 *v36; // rdi
  __int64 v37; // r12
  _BYTE *v38; // rdi
  unsigned __int64 v39; // rsi
  size_t v41; // r8
  _QWORD *v42; // r14
  _BYTE *v43; // r9
  size_t v44; // r10
  const char *v45; // rax
  __int64 *v46; // rax
  unsigned __int64 v47; // r14
  const char *v48; // rax
  unsigned __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // r14
  __int64 v52; // rax
  _QWORD *v53; // rdi
  _BYTE *src; // [rsp+10h] [rbp-130h]
  size_t n; // [rsp+18h] [rbp-128h]
  size_t na; // [rsp+18h] [rbp-128h]
  __int64 v57; // [rsp+28h] [rbp-118h]
  __int64 v58; // [rsp+28h] [rbp-118h]
  int v59; // [rsp+28h] [rbp-118h]
  size_t v60; // [rsp+28h] [rbp-118h]
  size_t v61; // [rsp+28h] [rbp-118h]
  __int64 v62; // [rsp+30h] [rbp-110h] BYREF
  __int64 v63; // [rsp+38h] [rbp-108h]
  const char *v64; // [rsp+40h] [rbp-100h] BYREF
  size_t v65; // [rsp+48h] [rbp-F8h]
  _QWORD v66[2]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v67; // [rsp+60h] [rbp-E0h]
  __int64 v68; // [rsp+68h] [rbp-D8h]
  __int64 v69; // [rsp+70h] [rbp-D0h]
  _BYTE *v70; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v71; // [rsp+88h] [rbp-B8h]
  _BYTE v72[176]; // [rsp+90h] [rbp-B0h] BYREF

  if ( *(_BYTE *)(a1 + 176) )
    return 0;
  v1 = *(_QWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 16);
  if ( v3 != v1 )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(v3 + 8);
      *(_QWORD *)(a1 + 16) = v4;
      v5 = *(_QWORD *)(v3 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v5 == v3 + 24 )
        break;
      if ( !v5 )
        break;
      v6 = v5 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
        break;
      v7 = *(_BYTE *)(v5 - 24);
      if ( v7 == 30 || v7 == 35 )
      {
        v37 = a1 + 32;
        v39 = sub_AA4E50(v3 - 24);
        if ( !v39 )
          v39 = v6;
        sub_D5F1F0(a1 + 32, v39);
        return v37;
      }
      v3 = v4;
      if ( v4 == v1 )
        goto LABEL_9;
    }
LABEL_62:
    BUG();
  }
LABEL_9:
  v8 = *(_BYTE *)(a1 + 177) == 0;
  *(_BYTE *)(a1 + 176) = 1;
  if ( v8 )
    return 0;
  if ( (unsigned __int8)sub_B2D610(*(_QWORD *)a1, 41) )
    return 0;
  v9 = *(_QWORD *)a1;
  v70 = v72;
  v10 = *(_QWORD *)(v9 + 80);
  v71 = 0x1000000000LL;
  v57 = v9 + 72;
  if ( v10 == v9 + 72 )
    return 0;
  do
  {
    if ( !v10 )
      BUG();
    v11 = *(_QWORD *)(v10 + 32);
    if ( v10 + 24 != v11 )
    {
      while ( v11 )
      {
        if ( *(_BYTE *)(v11 - 24) == 85
          && !(unsigned __int8)sub_A73ED0((_QWORD *)(v11 + 48), 41)
          && !(unsigned __int8)sub_B49560(v11 - 24, 41)
          && (*(_WORD *)(v11 - 22) & 3) != 2 )
        {
          v14 = (unsigned int)v71;
          v15 = (unsigned int)v71 + 1LL;
          if ( v15 > HIDWORD(v71) )
          {
            sub_C8D5F0((__int64)&v70, v72, v15, 8u, v12, v13);
            v14 = (unsigned int)v71;
          }
          *(_QWORD *)&v70[8 * v14] = v11 - 24;
          LODWORD(v71) = v71 + 1;
        }
        v11 = *(_QWORD *)(v11 + 8);
        if ( v10 + 24 == v11 )
          goto LABEL_23;
      }
      goto LABEL_62;
    }
LABEL_23:
    v10 = *(_QWORD *)(v10 + 8);
  }
  while ( v57 != v10 );
  if ( !(_DWORD)v71 )
  {
    v38 = v70;
    v37 = 0;
    goto LABEL_37;
  }
  v16 = sub_B2BE50(*(_QWORD *)a1);
  v17 = *(_QWORD *)a1;
  v18 = (__int64 *)v16;
  v19 = *(const char **)(a1 + 8);
  LOWORD(v67) = 257;
  if ( *v19 )
  {
    v64 = v19;
    LOBYTE(v67) = 3;
  }
  v20 = sub_22077B0(0x50u);
  v21 = v20;
  if ( v20 )
    sub_AA4D50(v20, (__int64)v18, (__int64)&v64, v17, 0);
  v22 = sub_BCB2D0(v18);
  v23 = (const char *)sub_BCE3C0(v18, 0);
  v24 = *(_QWORD **)v23;
  v64 = v23;
  v65 = v22;
  v25 = sub_BD0B90(v24, &v64, 2, 0);
  v26 = *(_QWORD *)a1;
  v27 = (__int64)v25;
  if ( (*(_BYTE *)(*(_QWORD *)a1 + 2LL) & 8) == 0 )
  {
    v41 = *(_QWORD *)(v26 + 40);
    v42 = *(_QWORD **)v41;
    v64 = (const char *)v66;
    v43 = *(_BYTE **)(v41 + 232);
    v44 = *(_QWORD *)(v41 + 240);
    if ( &v43[v44] && !v43 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v62 = *(_QWORD *)(v41 + 240);
    if ( v44 > 0xF )
    {
      src = v43;
      na = v44;
      v60 = v41;
      v52 = sub_22409D0((__int64)&v64, (unsigned __int64 *)&v62, 0);
      v41 = v60;
      v44 = na;
      v64 = (const char *)v52;
      v53 = (_QWORD *)v52;
      v43 = src;
      v66[0] = v62;
    }
    else
    {
      if ( v44 == 1 )
      {
        LOBYTE(v66[0]) = *v43;
        v45 = (const char *)v66;
LABEL_52:
        v65 = v44;
        v45[v44] = 0;
        v67 = *(_QWORD *)(v41 + 264);
        n = v41;
        v68 = *(_QWORD *)(v41 + 272);
        v69 = *(_QWORD *)(v41 + 280);
        v59 = sub_B2AC60(&v64);
        v46 = (__int64 *)sub_BCB2D0(v42);
        v47 = sub_BCF640(v46, 1u);
        v48 = sub_B2AB60(v59);
        sub_BA8CA0(n, (__int64)v48, v49, v47);
        v51 = v50;
        if ( v64 != (const char *)v66 )
          j_j___libc_free_0((unsigned __int64)v64);
        sub_B2E8C0(*(_QWORD *)a1, v51);
        v26 = *(_QWORD *)a1;
        goto LABEL_30;
      }
      if ( !v44 )
      {
        v45 = (const char *)v66;
        goto LABEL_52;
      }
      v53 = v66;
    }
    v61 = v41;
    memcpy(v53, v43, v44);
    v44 = v62;
    v45 = v64;
    v41 = v61;
    goto LABEL_52;
  }
LABEL_30:
  v28 = sub_B2E500(v26);
  v29 = sub_B2A630(v28);
  if ( v29 <= 10 )
  {
    if ( v29 <= 6 )
      goto LABEL_32;
LABEL_45:
    sub_C64ED0("Scoped EH not supported", 1u);
  }
  if ( v29 == 12 )
    goto LABEL_45;
LABEL_32:
  sub_B43C20((__int64)&v62, v21);
  v64 = "cleanup.lpad";
  LOWORD(v67) = 259;
  v30 = sub_B49060(v27, 1u, (__int64)&v64, v62, v63);
  *(_WORD *)(v30 + 2) |= 1u;
  v31 = v30;
  sub_B43C20((__int64)&v64, v21);
  v32 = v65;
  v58 = (__int64)v64;
  v33 = sub_BD2C40(72, 1u);
  v34 = (__int64)v33;
  if ( v33 )
    sub_B4BCC0((__int64)v33, v31, v58, v32);
  v35 = 8LL * (unsigned int)(v71 - 1);
  if ( (_DWORD)v71 )
  {
    do
    {
      v36 = *(unsigned __int8 **)&v70[v35];
      v35 -= 8;
      sub_F566B0(v36, v21, *(_QWORD *)(a1 + 184));
    }
    while ( v35 != -8 );
  }
  v37 = a1 + 32;
  sub_D5F1F0(a1 + 32, v34);
  v38 = v70;
LABEL_37:
  if ( v38 != v72 )
    _libc_free((unsigned __int64)v38);
  return v37;
}
