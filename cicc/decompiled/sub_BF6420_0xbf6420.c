// Function: sub_BF6420
// Address: 0xbf6420
//
__int64 __fastcall sub_BF6420(__int64 **a1, _BYTE *a2, const char *a3)
{
  unsigned __int8 v6; // cl
  unsigned int v7; // r15d
  bool v8; // di
  __int64 v9; // rsi
  char *v10; // r8
  char *v11; // rdx
  __int64 *v12; // rdi
  __int64 *v14; // rbx
  __int64 v15; // r12
  _BYTE *v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __int64 v19; // rdi
  _BYTE *v20; // rax
  char *v21; // r12
  _BYTE *v22; // rax
  char *v23; // r8
  char *v24; // r14
  int v25; // ecx
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // eax
  char v29; // cl
  char *v30; // rax
  char *v31; // rdx
  __int64 *v32; // rdi
  unsigned int v33; // eax
  _BYTE *v34; // rax
  _BYTE *v35; // rax
  _BYTE *v36; // rax
  __int64 v37; // rax
  __int64 *v38; // rdi
  const char *v39; // rax
  char v40; // dl
  unsigned __int64 v41; // rax
  char v42; // al
  int v43; // edx
  unsigned int v44; // ecx
  _QWORD *v45; // rax
  __int64 *v46; // rbx
  __int64 v47; // r12
  _QWORD *v48; // rax
  __int64 *v49; // rbx
  unsigned int v50; // ecx
  __int64 v51; // rax
  __int64 v52; // rax
  int v53; // eax
  int v54; // eax
  __int64 v55; // rax
  __int64 v56; // rax
  unsigned int v57; // r15d
  int v58; // eax
  __int64 *v59; // r15
  __int64 *v60; // rdi
  __int64 *v61; // rdi
  __int64 *v62; // rdi
  __int64 *v63; // r12
  __int64 *v64; // rdi
  unsigned int v65; // [rsp+14h] [rbp-ECh]
  unsigned __int64 v66; // [rsp+18h] [rbp-E8h]
  unsigned int v67; // [rsp+20h] [rbp-E0h]
  unsigned int v68; // [rsp+20h] [rbp-E0h]
  char *v69; // [rsp+20h] [rbp-E0h]
  char *v70; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v71; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v72; // [rsp+34h] [rbp-CCh]
  unsigned int v73; // [rsp+34h] [rbp-CCh]
  unsigned int v74; // [rsp+34h] [rbp-CCh]
  const char *v75[2]; // [rsp+38h] [rbp-C8h] BYREF
  _BYTE *v76; // [rsp+48h] [rbp-B8h] BYREF
  _QWORD *v77; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v78; // [rsp+58h] [rbp-A8h]
  _BYTE *v79[4]; // [rsp+60h] [rbp-A0h] BYREF
  char v80; // [rsp+80h] [rbp-80h]
  char v81; // [rsp+81h] [rbp-7Fh]
  const char *v82; // [rsp+90h] [rbp-70h] BYREF
  char *v83; // [rsp+98h] [rbp-68h]
  __int64 v84; // [rsp+A0h] [rbp-60h]
  int v85; // [rsp+A8h] [rbp-58h]
  char v86; // [rsp+ACh] [rbp-54h]
  char v87; // [rsp+B0h] [rbp-50h] BYREF
  char v88; // [rsp+B1h] [rbp-4Fh]

  v75[0] = a3;
  v6 = *(a3 - 16);
  if ( (v6 & 2) != 0 )
  {
    v7 = *((_DWORD *)a3 - 6);
    if ( v7 )
      goto LABEL_3;
LABEL_12:
    v14 = *a1;
    if ( !*a1 )
      return 0;
    v88 = 1;
    v82 = "TBAA metadata cannot have 0 operands";
    v87 = 3;
    v15 = *v14;
    if ( *v14 )
    {
      sub_CA0E80(&v82, *v14);
      v16 = *(_BYTE **)(v15 + 32);
      if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 24) )
      {
        sub_CB5D20(v15, 10);
      }
      else
      {
        *(_QWORD *)(v15 + 32) = v16 + 1;
        *v16 = 10;
      }
      v15 = *v14;
    }
    *((_BYTE *)v14 + 152) = 1;
    if ( !v15 )
      return 0;
    if ( *a2 <= 0x1Cu )
    {
      sub_A5C020(a2, v15, 1, (__int64)(v14 + 2));
      v17 = *v14;
      v18 = *(_BYTE **)(*v14 + 32);
      if ( (unsigned __int64)v18 < *(_QWORD *)(*v14 + 24) )
        goto LABEL_20;
    }
    else
    {
      sub_A693B0((__int64)a2, (_BYTE *)v15, (__int64)(v14 + 2), 0);
      v17 = *v14;
      v18 = *(_BYTE **)(*v14 + 32);
      if ( (unsigned __int64)v18 < *(_QWORD *)(*v14 + 24) )
      {
LABEL_20:
        *(_QWORD *)(v17 + 32) = v18 + 1;
        *v18 = 10;
        goto LABEL_21;
      }
    }
    sub_CB5D20(v17, 10);
LABEL_21:
    if ( v75[0] )
    {
      sub_A62C00(v75[0], *v14, (__int64)(v14 + 2), v14[1]);
      v19 = *v14;
      v20 = *(_BYTE **)(*v14 + 32);
      if ( (unsigned __int64)v20 >= *(_QWORD *)(*v14 + 24) )
      {
        sub_CB5D20(v19, 10);
      }
      else
      {
        *(_QWORD *)(v19 + 32) = v20 + 1;
        *v20 = 10;
      }
    }
    return 0;
  }
  v7 = (*((_WORD *)a3 - 8) >> 6) & 0xF;
  if ( !v7 )
    goto LABEL_12;
LABEL_3:
  v8 = ((*a2 - 85) & 0xFB) != 0;
  v9 = (unsigned __int8)(*a2 - 61) & 0xFA;
  LOBYTE(v9) = ((*a2 - 61) & 0xFA) != 0;
  if ( ((unsigned __int8)v9 & v8) != 0 )
  {
    v61 = *a1;
    v79[0] = a2;
    if ( v61 )
    {
      v88 = 1;
      v82 = "This instruction shall not have a TBAA access tag!";
      v87 = 3;
      sub_BE0C10(v61, (__int64)&v82, v79);
    }
    return 0;
  }
  v10 = (char *)(a3 - 16);
  if ( (v6 & 2) != 0 )
    v11 = (char *)*((_QWORD *)a3 - 4);
  else
    v11 = &v10[-8 * ((v6 >> 2) & 0xF)];
  v72 = v7 > 2 && (unsigned __int8)(**(_BYTE **)v11 - 5) <= 0x1Fu;
  if ( !v72 )
  {
    v12 = *a1;
    v79[0] = a2;
    if ( v12 )
    {
      v88 = 1;
      v82 = "Old-style TBAA is no longer allowed, use struct-path TBAA instead";
      v87 = 3;
      sub_BE0C10(v12, (__int64)&v82, v79);
    }
    return v72;
  }
  v70 = (char *)(a3 - 16);
  v21 = *(char **)sub_A17150(v10);
  if ( v21 && (unsigned __int8)(*v21 - 5) >= 0x20u )
    v21 = 0;
  v22 = sub_A17150(v70);
  v23 = v70;
  v24 = (char *)*((_QWORD *)v22 + 1);
  if ( !v24 )
    goto LABEL_31;
  if ( (unsigned __int8)(*v24 - 5) > 0x1Fu )
  {
    v24 = 0;
LABEL_31:
    v72 = 0;
    v25 = 3;
    if ( v7 <= 4 )
      goto LABEL_32;
    v38 = *a1;
    v79[0] = a2;
    if ( !v38 )
      return 0;
    v88 = 1;
    v39 = "Struct tag metadata must have either 3 or 4 operands";
LABEL_69:
    v82 = v39;
    v87 = 3;
    sub_BF02E0(v38, (__int64)&v82, v79, v75);
    return 0;
  }
  if ( (*(v24 - 16) & 2) != 0 )
    v33 = *((_DWORD *)v24 - 6);
  else
    v33 = (*((_WORD *)v24 - 8) >> 6) & 0xF;
  if ( v33 <= 2 )
    goto LABEL_31;
  v34 = sub_A17150(v24 - 16);
  v23 = v70;
  v35 = *(_BYTE **)v34;
  if ( !v35 || (unsigned __int8)(*v35 - 5) > 0x1Fu )
    goto LABEL_31;
  if ( v7 - 4 > 1 )
  {
    v38 = *a1;
    v79[0] = a2;
    if ( !v38 )
      return 0;
    v88 = 1;
    v39 = "Access tag metadata must have either 4 or 5 operands";
    goto LABEL_69;
  }
  v36 = sub_A17150(v70);
  v23 = v70;
  v37 = *((_QWORD *)v36 + 3);
  if ( !v37 || *(_BYTE *)v37 != 1 || (v25 = 4, **(_BYTE **)(v37 + 136) != 17) )
  {
    v38 = *a1;
    v79[0] = a2;
    if ( !v38 )
      return 0;
    v88 = 1;
    v39 = "Access size field must be a constant";
    goto LABEL_69;
  }
LABEL_32:
  if ( (v25 != 3) + 4 != v7 )
    goto LABEL_33;
  v69 = v23;
  v55 = *(_QWORD *)&sub_A17150(v23)[8 * v25];
  if ( !v55 || *(_BYTE *)v55 != 1 || (v56 = *(_QWORD *)(v55 + 136), *(_BYTE *)v56 != 17) )
  {
    v38 = *a1;
    v79[0] = a2;
    if ( !v38 )
      return 0;
    v88 = 1;
    v39 = "Immutability tag on struct tag metadata must be a constant";
    goto LABEL_69;
  }
  v57 = *(_DWORD *)(v56 + 32);
  v23 = v69;
  if ( v57 <= 0x40 )
  {
    if ( *(_QWORD *)(v56 + 24) >= 2u )
    {
LABEL_113:
      v38 = *a1;
      v79[0] = a2;
      if ( !v38 )
        return 0;
      v88 = 1;
      v39 = "Immutability part of the struct tag metadata must be either 0 or 1";
      goto LABEL_69;
    }
  }
  else
  {
    v58 = sub_C444A0(v56 + 24);
    v23 = v69;
    if ( v57 != v58 && v58 != v57 - 1 )
      goto LABEL_113;
  }
LABEL_33:
  if ( v24 == 0 || v21 == 0 )
  {
    v59 = *a1;
    if ( *a1 )
    {
      v60 = *a1;
      v88 = 1;
      v82 = "Malformed struct tag metadata: base and access-type should be non-null and point to Metadata nodes";
      v87 = 3;
      sub_BDBF70(v60, (__int64)&v82);
      if ( *v59 )
      {
        sub_BDBD80((__int64)v59, a2);
        if ( v75[0] )
          sub_BD9900(v59, v75[0]);
        if ( v21 )
        {
          sub_BD9900(v59, v21);
        }
        else if ( v24 )
        {
          sub_BD9900(v59, v24);
        }
      }
    }
    return 0;
  }
  if ( !v72 )
  {
    v9 = (__int64)v24;
    if ( !(unsigned __int8)sub_BF5560((__int64)a1, v24) )
    {
      v63 = *a1;
      if ( *a1 )
      {
        v64 = *a1;
        v88 = 1;
        v82 = "Access type node must be a valid scalar type";
        v87 = 3;
        sub_BDBF70(v64, (__int64)&v82);
        if ( *v63 )
        {
          sub_BDBD80((__int64)v63, a2);
          if ( v75[0] )
            sub_BD9900(v63, v75[0]);
          sub_BD9900(v63, v24);
        }
      }
      return 0;
    }
    v23 = (char *)(v75[0] - 16);
  }
  v26 = *((_QWORD *)sub_A17150(v23) + 2);
  if ( !v26 || *(_BYTE *)v26 != 1 || (v27 = *(_QWORD *)(v26 + 136), *(_BYTE *)v27 != 17) )
  {
    v38 = *a1;
    v79[0] = a2;
    if ( !v38 )
      return 0;
    v88 = 1;
    v39 = "Offset must be constant integer";
    goto LABEL_69;
  }
  v78 = *(_DWORD *)(v27 + 32);
  if ( v78 > 0x40 )
  {
    v9 = v27 + 24;
    sub_C43780(&v77, v27 + 24);
  }
  else
  {
    v77 = *(_QWORD **)(v27 + 24);
  }
  v82 = 0;
  v83 = &v87;
  v84 = 4;
  v85 = 0;
  v86 = 1;
  v71 = 0;
  do
  {
    v28 = (*(v21 - 16) & 2) != 0 ? *((_DWORD *)v21 - 6) : (*((_WORD *)v21 - 8) >> 6) & 0xF;
    if ( v28 <= 1 )
      break;
    v29 = v86;
    if ( !v86 )
      goto LABEL_71;
    v30 = v83;
    v9 = HIDWORD(v84);
    v31 = &v83[8 * HIDWORD(v84)];
    if ( v83 != v31 )
    {
      while ( v21 != *(char **)v30 )
      {
        v30 += 8;
        if ( v31 == v30 )
          goto LABEL_101;
      }
LABEL_49:
      v32 = *a1;
      v76 = a2;
      if ( v32 )
      {
        v81 = 1;
        v9 = (__int64)v79;
        v79[0] = "Cycle detected in struct path";
        v80 = 3;
        sub_BF02E0(v32, (__int64)v79, &v76, v75);
LABEL_51:
        v29 = v86;
      }
      v72 = 0;
      goto LABEL_53;
    }
LABEL_101:
    if ( HIDWORD(v84) < (unsigned int)v84 )
    {
      ++HIDWORD(v84);
      *(_QWORD *)v31 = v21;
      ++v82;
    }
    else
    {
LABEL_71:
      v9 = (__int64)v21;
      sub_C8CC70(&v82, v21);
      v29 = v86;
      if ( !v40 )
        goto LABEL_49;
    }
    v9 = (__int64)a2;
    v41 = sub_BF6140((__int64)a1, a2, v21, v72);
    v65 = HIDWORD(v41);
    if ( (_BYTE)v41 )
      goto LABEL_51;
    v9 = (__int64)v21;
    v66 = HIDWORD(v41);
    v71 |= v24 == v21;
    v42 = sub_BF5560((__int64)a1, v21);
    v43 = v66;
    v44 = v78;
    if ( v24 == v21 || v42 )
    {
      if ( v78 > 0x40 )
      {
        v67 = v78;
        v53 = sub_C444A0(&v77);
        v44 = v67;
        v43 = v66;
        if ( v67 - v53 > 0x40 )
        {
LABEL_78:
          v46 = *a1;
          if ( v46 )
          {
            v9 = (__int64)v79;
            v81 = 1;
            v79[0] = "Offset not zero at the point of scalar access";
            v80 = 3;
            sub_BDBF70(v46, (__int64)v79);
            if ( *v46 )
            {
              sub_BDBD80((__int64)v46, a2);
              if ( v75[0] )
                sub_BD9900(v46, v75[0]);
              v47 = *v46;
              sub_C49420(&v77, *v46, 1);
              v9 = 10;
              sub_A51310(v47, 0xAu);
            }
          }
LABEL_93:
          v29 = v86;
          v72 = 0;
          goto LABEL_53;
        }
        v45 = (_QWORD *)*v77;
      }
      else
      {
        v45 = v77;
      }
      if ( v45 )
        goto LABEL_78;
    }
    if ( v43 != v44 )
    {
      if ( v43 )
      {
        if ( v43 != -1 || v72 != 1 )
        {
LABEL_88:
          v49 = *a1;
          v73 = v44;
          if ( v49 )
          {
            v9 = (__int64)v79;
            v81 = 1;
            v79[0] = "Access bit-width not the same as description bit-width";
            v80 = 3;
            sub_BDBF70(v49, (__int64)v79);
            if ( *v49 )
            {
              sub_BDBD80((__int64)v49, a2);
              v50 = v73;
              if ( v75[0] )
              {
                sub_BD9900(v49, v75[0]);
                v50 = v73;
              }
              v74 = v50;
              v51 = sub_CB59D0(*v49, v65);
              sub_A51310(v51, 0xAu);
              v52 = sub_CB59D0(*v49, v74);
              v9 = 10;
              sub_A51310(v52, 0xAu);
            }
          }
          goto LABEL_93;
        }
      }
      else
      {
        if ( v44 > 0x40 )
        {
          v68 = v44;
          v54 = sub_C444A0(&v77);
          v44 = v68;
          if ( v68 - v54 > 0x40 )
            goto LABEL_88;
          v48 = (_QWORD *)*v77;
        }
        else
        {
          v48 = v77;
        }
        if ( v48 )
          goto LABEL_88;
      }
    }
    if ( (v72 & v71) != 0 )
    {
      v72 &= v71;
      v29 = v86;
      goto LABEL_53;
    }
    v9 = (__int64)a2;
    v21 = (char *)sub_BE0500(a1, a2, v21, (__int64)&v77, v72);
  }
  while ( v21 );
  v72 = v71;
  if ( !v71 )
  {
    v62 = *a1;
    v76 = a2;
    if ( v62 )
    {
      v81 = 1;
      v9 = (__int64)v79;
      v79[0] = "Did not see access type in access path!";
      v80 = 3;
      sub_BF02E0(v62, (__int64)v79, &v76, v75);
    }
  }
  v29 = v86;
LABEL_53:
  if ( !v29 )
    _libc_free(v83, v9);
  if ( v78 > 0x40 && v77 )
    j_j___libc_free_0_0(v77);
  return v72;
}
