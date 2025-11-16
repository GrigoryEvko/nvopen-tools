// Function: sub_16635B0
// Address: 0x16635b0
//
__int64 __fastcall sub_16635B0(__int64 **a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5, unsigned int a6)
{
  char v8; // al
  unsigned __int8 *v9; // r14
  __int64 v10; // rcx
  unsigned int v11; // edx
  __int64 *v12; // r15
  unsigned int v13; // r13d
  __int64 *v14; // rdi
  __int64 v16; // rsi
  __int64 *v17; // r13
  __int64 *v18; // rdi
  __int64 v19; // rax
  _BYTE *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  __int64 **v24; // r8
  char v25; // dl
  unsigned __int64 v26; // rax
  int v27; // ecx
  char v28; // al
  int v29; // edx
  unsigned int v30; // ecx
  _QWORD *v31; // rax
  __int64 *v32; // rbx
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // eax
  _QWORD *v39; // rax
  __int64 *v40; // rbx
  unsigned int v41; // ecx
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // eax
  __int64 **v45; // rsi
  __int64 **v46; // rax
  __int64 **v47; // rcx
  __int64 *v48; // r9
  int v49; // eax
  int v50; // eax
  __int64 *v51; // rdi
  const char *v52; // rax
  __int64 *v53; // rdi
  __int64 v54; // rax
  __int64 *v55; // rdi
  __int64 *v56; // r14
  __int64 *v57; // rdi
  __int64 *v58; // rdi
  unsigned int v59; // [rsp+14h] [rbp-ECh]
  unsigned __int8 *v60; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v61; // [rsp+20h] [rbp-E0h]
  __int64 v62; // [rsp+28h] [rbp-D8h]
  unsigned int v63; // [rsp+28h] [rbp-D8h]
  unsigned int v64; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v65; // [rsp+34h] [rbp-CCh]
  unsigned int v66; // [rsp+34h] [rbp-CCh]
  unsigned int v67; // [rsp+34h] [rbp-CCh]
  unsigned __int8 *v68[2]; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v69; // [rsp+48h] [rbp-B8h] BYREF
  _QWORD *v70; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v71; // [rsp+58h] [rbp-A8h]
  __int64 v72[2]; // [rsp+60h] [rbp-A0h] BYREF
  char v73; // [rsp+70h] [rbp-90h]
  char v74; // [rsp+71h] [rbp-8Fh]
  const char *v75; // [rsp+80h] [rbp-80h] BYREF
  __int64 **v76; // [rsp+88h] [rbp-78h]
  __int64 **v77; // [rsp+90h] [rbp-70h]
  __int64 v78; // [rsp+98h] [rbp-68h]
  int v79; // [rsp+A0h] [rbp-60h]
  _BYTE v80[88]; // [rsp+A8h] [rbp-58h] BYREF

  v8 = *(_BYTE *)(a2 + 16);
  v68[0] = a3;
  if ( ((v8 - 78) & 0xFB) != 0 && ((v8 - 54) & 0xFA) != 0 )
  {
    v53 = *a1;
    v72[0] = a2;
    if ( v53 )
    {
      v75 = "This instruction shall not have a TBAA access tag!";
      LOWORD(v77) = 259;
      sub_1654980(v53, (__int64)&v75, v72);
    }
    return 0;
  }
  v9 = v68[0];
  v10 = *((unsigned int *)v68[0] + 2);
  v11 = *((_DWORD *)v68[0] + 2);
  v12 = *(__int64 **)&v68[0][-8 * v10];
  LOBYTE(a6) = (unsigned int)v10 > 2 && (unsigned __int8)(*(_BYTE *)v12 - 4) <= 0x1Eu;
  v13 = a6;
  if ( !(_BYTE)a6 )
  {
    v14 = *a1;
    v72[0] = a2;
    if ( v14 )
    {
      v75 = "Old-style TBAA is no longer allowed, use struct-path TBAA instead";
      LOWORD(v77) = 259;
      sub_1654980(v14, (__int64)&v75, v72);
    }
    return v13;
  }
  v16 = *(_QWORD *)&v68[0][8 * (1 - v10)];
  v60 = (unsigned __int8 *)v16;
  if ( !v16 )
  {
    if ( (unsigned int)v10 > 4 )
      goto LABEL_113;
    if ( (_DWORD)v10 != 4 )
      goto LABEL_10;
    goto LABEL_112;
  }
  if ( (unsigned __int8)(*(_BYTE *)v16 - 4) > 0x1Eu )
  {
    if ( (unsigned int)v10 <= 4 )
    {
      if ( (_DWORD)v10 != 4 )
      {
LABEL_10:
        v17 = *a1;
        if ( *a1 )
        {
          v18 = *a1;
          v75 = "Malformed struct tag metadata: base and access-type should be non-null and point to Metadata nodes";
          LOWORD(v77) = 259;
          sub_164FF40(v18, (__int64)&v75);
          if ( *v17 )
          {
            sub_164FA80(v17, a2);
            if ( v68[0] )
              sub_164ED40(v17, v68[0]);
            sub_164ED40(v17, (unsigned __int8 *)v12);
          }
        }
        return 0;
      }
      v65 = 0;
      v35 = 3;
      v60 = 0;
      goto LABEL_49;
    }
    goto LABEL_113;
  }
  v19 = *(unsigned int *)(v16 + 8);
  if ( (unsigned int)v19 > 2 )
  {
    v20 = *(_BYTE **)(v16 - 8 * v19);
    if ( v20 )
    {
      if ( (unsigned __int8)(*v20 - 4) <= 0x1Eu )
      {
        if ( (unsigned int)(v10 - 4) > 1 )
        {
          v51 = *a1;
          v72[0] = a2;
          if ( !v51 )
            return 0;
          BYTE1(v77) = 1;
          v52 = "Access tag metadata must have either 4 or 5 operands";
          goto LABEL_102;
        }
        v34 = *(_QWORD *)&v68[0][8 * (3 - v10)];
        if ( !v34 || *(_BYTE *)v34 != 1 || *(_BYTE *)(*(_QWORD *)(v34 + 136) + 16LL) != 13 )
        {
          v51 = *a1;
          v72[0] = a2;
          if ( !v51 )
            return 0;
          BYTE1(v77) = 1;
          v52 = "Access size field must be a constant";
          goto LABEL_102;
        }
        v65 = (unsigned int)v10 > 2 && (unsigned __int8)(*(_BYTE *)v12 - 4) <= 0x1Eu;
        if ( v11 != 5 )
          goto LABEL_23;
        v35 = 4;
LABEL_49:
        v36 = *(_QWORD *)&v68[0][8 * (v35 - v10)];
        if ( !v36 || *(_BYTE *)v36 != 1 || (v37 = *(_QWORD *)(v36 + 136), *(_BYTE *)(v37 + 16) != 13) )
        {
          v51 = *a1;
          v72[0] = a2;
          if ( !v51 )
            return 0;
          BYTE1(v77) = 1;
          v52 = "Immutability tag on struct tag metadata must be a constant";
          goto LABEL_102;
        }
        v13 = *(_DWORD *)(v37 + 32);
        if ( v13 <= 0x40 )
        {
          v54 = *(_QWORD *)(v37 + 24);
          if ( !v54 )
            goto LABEL_56;
          LOBYTE(v13) = v54 == 1;
        }
        else
        {
          v62 = *((unsigned int *)v68[0] + 2);
          v38 = sub_16A57B0(v37 + 24);
          v10 = v62;
          if ( v13 == v38 )
            goto LABEL_56;
          LOBYTE(v13) = v13 - 1 == v38;
        }
        if ( !(_BYTE)v13 )
        {
          v58 = *a1;
          v72[0] = a2;
          if ( v58 )
          {
            v75 = "Immutability part of the struct tag metadata must be either 0 or 1";
            LOWORD(v77) = 259;
            sub_165C320(v58, (__int64)&v75, v72, v68);
          }
          return v13;
        }
LABEL_56:
        if ( !v60 )
          goto LABEL_10;
        if ( v65 )
          goto LABEL_23;
        goto LABEL_21;
      }
    }
  }
  if ( v11 > 4 )
  {
LABEL_113:
    v51 = *a1;
    v72[0] = a2;
    if ( !v51 )
      return 0;
    BYTE1(v77) = 1;
    v52 = "Struct tag metadata must have either 3 or 4 operands";
    goto LABEL_102;
  }
  if ( v11 == 4 )
  {
LABEL_112:
    v65 = 0;
    v35 = 3;
    goto LABEL_49;
  }
LABEL_21:
  v13 = sub_1662870((__int64)a1, (__int64)v60);
  if ( !(_BYTE)v13 )
  {
    v56 = *a1;
    if ( *a1 )
    {
      v57 = *a1;
      v75 = "Access type node must be a valid scalar type";
      LOWORD(v77) = 259;
      sub_164FF40(v57, (__int64)&v75);
      if ( *v56 )
      {
        sub_164FA80(v56, a2);
        if ( v68[0] )
          sub_164ED40(v56, v68[0]);
        sub_164ED40(v56, v60);
      }
      return v13;
    }
    return 0;
  }
  v9 = v68[0];
  v65 = 0;
  v10 = *((unsigned int *)v68[0] + 2);
LABEL_23:
  v21 = *(_QWORD *)&v9[8 * (2 - v10)];
  if ( !v21 || *(_BYTE *)v21 != 1 || (v22 = *(_QWORD *)(v21 + 136), *(_BYTE *)(v22 + 16) != 13) )
  {
    v51 = *a1;
    v72[0] = a2;
    if ( v51 )
    {
      BYTE1(v77) = 1;
      v52 = "Offset must be constant integer";
LABEL_102:
      v75 = v52;
      LOBYTE(v77) = 3;
      sub_165C320(v51, (__int64)&v75, v72, v68);
    }
    return 0;
  }
  v71 = *(_DWORD *)(v22 + 32);
  if ( v71 > 0x40 )
    sub_16A4FD0(&v70, v22 + 24);
  else
    v70 = *(_QWORD **)(v22 + 24);
  v75 = 0;
  v13 = 0;
  v76 = (__int64 **)v80;
  v77 = (__int64 **)v80;
  v78 = 4;
  v79 = 0;
  while ( 1 )
  {
    if ( *((_DWORD *)v12 + 2) <= 1u )
    {
LABEL_74:
      if ( !(_BYTE)v13 )
      {
        v55 = *a1;
        v69 = a2;
        if ( v55 )
        {
          v74 = 1;
          v72[0] = (__int64)"Did not see access type in access path!";
          v73 = 3;
          sub_165C320(v55, (__int64)v72, &v69, v68);
        }
      }
      v23 = (unsigned __int64)v77;
      v24 = v76;
      goto LABEL_86;
    }
    v23 = (unsigned __int64)v77;
    v24 = v76;
    if ( v77 != v76 )
      goto LABEL_31;
    v45 = &v77[HIDWORD(v78)];
    if ( v77 != v45 )
      break;
LABEL_97:
    if ( HIDWORD(v78) < (unsigned int)v78 )
    {
      ++HIDWORD(v78);
      *v45 = v12;
      ++v75;
      goto LABEL_32;
    }
LABEL_31:
    sub_16CCBA0(&v75, v12);
    v23 = (unsigned __int64)v77;
    v24 = v76;
    if ( !v25 )
      goto LABEL_82;
LABEL_32:
    v26 = sub_1663300((__int64)a1, a2, (__int64)v12, v65);
    v59 = HIDWORD(v26);
    if ( (_BYTE)v26 )
      goto LABEL_84;
    v61 = HIDWORD(v26);
    LOBYTE(v27) = v60 == (unsigned __int8 *)v12;
    v13 |= v27;
    v28 = sub_1662870((__int64)a1, (__int64)v12);
    v29 = v61;
    v30 = v71;
    if ( v60 == (unsigned __int8 *)v12 || v28 )
    {
      if ( v71 > 0x40 )
      {
        v63 = v71;
        v49 = sub_16A57B0(&v70);
        v30 = v63;
        v29 = v61;
        if ( v63 - v49 > 0x40 )
        {
LABEL_38:
          v32 = *a1;
          if ( v32 )
          {
            v74 = 1;
            v72[0] = (__int64)"Offset not zero at the point of scalar access";
            v73 = 3;
            sub_164FF40(v32, (__int64)v72);
            if ( *v32 )
            {
              sub_164FA80(v32, a2);
              if ( v68[0] )
                sub_164ED40(v32, v68[0]);
              v33 = *v32;
              sub_16A95F0(&v70, *v32, 1);
              sub_1549FC0(v33, 0xAu);
            }
          }
LABEL_69:
          v23 = (unsigned __int64)v77;
          v24 = v76;
          v13 = 0;
          goto LABEL_86;
        }
        v31 = (_QWORD *)*v70;
      }
      else
      {
        v31 = v70;
      }
      if ( v31 )
        goto LABEL_38;
    }
    if ( v30 != v29 )
    {
      if ( v29 )
      {
        if ( v65 != 1 || v29 != -1 )
        {
LABEL_64:
          v40 = *a1;
          v66 = v30;
          if ( v40 )
          {
            v74 = 1;
            v72[0] = (__int64)"Access bit-width not the same as description bit-width";
            v73 = 3;
            sub_164FF40(v40, (__int64)v72);
            if ( *v40 )
            {
              sub_164FA80(v40, a2);
              v41 = v66;
              if ( v68[0] )
              {
                sub_164ED40(v40, v68[0]);
                v41 = v66;
              }
              v67 = v41;
              v42 = sub_16E7A90(*v40, v59);
              sub_1549FC0(v42, 0xAu);
              v43 = sub_16E7A90(*v40, v67);
              sub_1549FC0(v43, 0xAu);
            }
          }
          goto LABEL_69;
        }
      }
      else
      {
        if ( v30 > 0x40 )
        {
          v64 = v30;
          v50 = sub_16A57B0(&v70);
          v30 = v64;
          if ( v64 - v50 > 0x40 )
            goto LABEL_64;
          v39 = (_QWORD *)*v70;
        }
        else
        {
          v39 = v70;
        }
        if ( v39 )
          goto LABEL_64;
      }
    }
    v44 = v65;
    if ( ((unsigned __int8)v13 & v65) != 0 )
    {
      v23 = (unsigned __int64)v77;
      v24 = v76;
      LOBYTE(v44) = v13 & v65;
      v13 = v44;
      goto LABEL_86;
    }
    v12 = sub_16544A0(a1, a2, (__int64)v12, (__int64)&v70, v65);
    if ( !v12 )
      goto LABEL_74;
  }
  v46 = v77;
  v47 = 0;
  while ( v12 != *v46 )
  {
    if ( *v46 == (__int64 *)-2LL )
      v47 = v46;
    if ( v45 == ++v46 )
    {
      if ( !v47 )
        goto LABEL_97;
      *v47 = v12;
      --v79;
      ++v75;
      goto LABEL_32;
    }
  }
LABEL_82:
  v48 = *a1;
  v69 = a2;
  if ( v48 )
  {
    v74 = 1;
    v72[0] = (__int64)"Cycle detected in struct path";
    v73 = 3;
    sub_165C320(v48, (__int64)v72, &v69, v68);
LABEL_84:
    v23 = (unsigned __int64)v77;
    v24 = v76;
  }
  v13 = 0;
LABEL_86:
  if ( (__int64 **)v23 != v24 )
    _libc_free(v23);
  if ( v71 > 0x40 && v70 )
    j_j___libc_free_0_0(v70);
  return v13;
}
