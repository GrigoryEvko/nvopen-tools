// Function: sub_1B481C0
// Address: 0x1b481c0
//
__int64 __fastcall sub_1B481C0(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v3; // r12d
  __int64 v4; // r13
  __int64 v6; // r14
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rax
  int v13; // eax
  unsigned int v14; // r14d
  _BYTE *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // r12
  unsigned __int64 v19; // rax
  __int64 v20; // r15
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rbx
  __int64 v24; // rsi
  _QWORD *v25; // rcx
  unsigned int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // r10
  unsigned int v29; // eax
  __int64 v30; // rbx
  char v31; // al
  char v32; // al
  char v33; // al
  char v34; // al
  __int64 v35; // r10
  __int64 v36; // rsi
  __int64 v37; // rbx
  _QWORD *v38; // rax
  __int64 v39; // r15
  __int64 v40; // rax
  __int64 *v41; // rbx
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rsi
  __int64 v45; // rdi
  unsigned __int64 v46; // rax
  __int64 v47; // rbx
  _QWORD *v48; // rax
  __int64 v49; // r15
  _QWORD *v50; // rax
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 *v53; // rbx
  __int64 v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rax
  unsigned __int64 v57; // [rsp+8h] [rbp-118h]
  unsigned __int64 v58; // [rsp+8h] [rbp-118h]
  unsigned __int64 v60; // [rsp+28h] [rbp-F8h]
  __int64 v61; // [rsp+28h] [rbp-F8h]
  __int64 v62; // [rsp+28h] [rbp-F8h]
  __int64 v63; // [rsp+28h] [rbp-F8h]
  __int64 v64; // [rsp+28h] [rbp-F8h]
  __int64 v65; // [rsp+28h] [rbp-F8h]
  __int64 v66[2]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v67; // [rsp+40h] [rbp-E0h]
  _BYTE *v68; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v69; // [rsp+58h] [rbp-C8h]
  _BYTE v70[64]; // [rsp+60h] [rbp-C0h] BYREF
  _BYTE *v71; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v72; // [rsp+A8h] [rbp-78h]
  _BYTE v73[112]; // [rsp+B0h] [rbp-70h] BYREF

  v3 = 0;
  v4 = *(_QWORD *)(a2 + 40);
  if ( (unsigned int)*(unsigned __int8 *)(sub_157ED60(v4) + 16) - 25 > 9 )
    return v3;
  v68 = v70;
  v69 = 0x800000000LL;
  v72 = 0x800000000LL;
  v6 = *(_QWORD *)(v4 + 8);
  v71 = v73;
  if ( !v6 )
    goto LABEL_29;
  while ( 1 )
  {
    v7 = sub_1648700(v6);
    if ( (unsigned __int8)(*((_BYTE *)v7 + 16) - 25) <= 9u )
      break;
    v6 = *(_QWORD *)(v6 + 8);
    if ( !v6 )
    {
      v3 = 0;
      goto LABEL_29;
    }
  }
LABEL_11:
  v9 = v7[5];
  v46 = sub_157EBA0(v9);
  if ( *(_BYTE *)(v46 + 16) != 26 )
    goto LABEL_9;
  if ( (*(_DWORD *)(v46 + 20) & 0xFFFFFFF) == 1 )
  {
    v12 = (unsigned int)v69;
    if ( (unsigned int)v69 >= HIDWORD(v69) )
    {
      sub_16CD150((__int64)&v68, v70, 0, 8, v10, v11);
      v12 = (unsigned int)v69;
    }
    *(_QWORD *)&v68[8 * v12] = v9;
    LODWORD(v69) = v69 + 1;
    v6 = *(_QWORD *)(v6 + 8);
    if ( v6 )
      goto LABEL_10;
  }
  else
  {
    v8 = (unsigned int)v72;
    if ( (unsigned int)v72 >= HIDWORD(v72) )
    {
      v58 = v46;
      sub_16CD150((__int64)&v71, v73, 0, 8, v10, v11);
      v8 = (unsigned int)v72;
      v46 = v58;
    }
    *(_QWORD *)&v71[8 * v8] = v46;
    LODWORD(v72) = v72 + 1;
LABEL_9:
    while ( 1 )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        break;
LABEL_10:
      v7 = sub_1648700(v6);
      if ( (unsigned __int8)(*((_BYTE *)v7 + 16) - 25) <= 9u )
        goto LABEL_11;
    }
  }
  v13 = v69;
  if ( (_DWORD)v69 )
  {
    v3 = (unsigned __int8)byte_4FB75A0;
    if ( byte_4FB75A0 )
    {
      do
      {
        v22 = *(_QWORD *)&v68[8 * v13 - 8];
        LODWORD(v69) = v13 - 1;
        sub_1AA6640(a2, v4, v22);
        v13 = v69;
      }
      while ( (_DWORD)v69 );
      v23 = *(_QWORD *)(v4 + 8);
      if ( v23 )
      {
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v23) + 16) - 25) > 9u )
        {
          v23 = *(_QWORD *)(v23 + 8);
          if ( !v23 )
            goto LABEL_61;
        }
      }
      else
      {
LABEL_61:
        sub_157F980(v4);
        v45 = *(_QWORD *)(a1 + 32);
        if ( v45 )
          sub_1B44390(v45, v4);
      }
      v15 = v71;
      goto LABEL_27;
    }
  }
  v14 = v72;
  v15 = v71;
  if ( !(_DWORD)v72 )
  {
LABEL_26:
    v3 = 0;
    goto LABEL_27;
  }
  while ( 1 )
  {
    while ( 1 )
    {
      v16 = v14--;
      v17 = *(_QWORD *)&v15[8 * v16 - 8];
      LODWORD(v72) = v14;
      v18 = *(_QWORD *)(v17 - 24);
      v19 = sub_157EBA0(v18);
      if ( *(_BYTE *)(v19 + 16) == 25 )
      {
        v20 = *(_QWORD *)(v17 - 48);
        v60 = v19;
        v21 = sub_157EBA0(v20);
        if ( *(_BYTE *)(v21 + 16) == 25 )
          break;
      }
      if ( !v14 )
        goto LABEL_26;
    }
    v57 = v21;
    if ( (unsigned int)*(unsigned __int8 *)(sub_157ED60(v18) + 16) - 25 > 9
      || (unsigned int)*(unsigned __int8 *)(sub_157ED60(v20) + 16) - 25 > 9 )
    {
      goto LABEL_25;
    }
    v24 = v17;
    sub_17050D0(a3, v17);
    v25 = (_QWORD *)v60;
    v26 = *(_DWORD *)(v57 + 20) & 0xFFFFFFF;
    if ( !v26 )
    {
      sub_157F2D0(v18, *(_QWORD *)(v17 + 40), 0);
      sub_157F2D0(v20, *(_QWORD *)(v17 + 40), 0);
      v67 = 257;
      v49 = a3[3];
      v50 = sub_1648A60(56, 0);
      v51 = (__int64)v50;
      if ( v50 )
        sub_15F6F90((__int64)v50, v49, 0, 0);
      v52 = a3[1];
      if ( v52 )
      {
        v53 = (__int64 *)a3[2];
        sub_157E9D0(v52 + 40, v51);
        v54 = *(_QWORD *)(v51 + 24);
        v55 = *v53;
        *(_QWORD *)(v51 + 32) = v53;
        v55 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v51 + 24) = v55 | v54 & 7;
        *(_QWORD *)(v55 + 8) = v51 + 24;
        *v53 = *v53 & 7 | (v51 + 24);
      }
      sub_164B780(v51, v66);
      v44 = v51;
      goto LABEL_59;
    }
    v27 = 4LL * v26;
    v28 = *(_QWORD *)(v57 - 24LL * v26);
    v29 = *(_DWORD *)(v60 + 20) & 0xFFFFFFF;
    if ( !v29 || (v27 = 4LL * v29, (v30 = *(_QWORD *)(v60 - 24LL * v29)) == 0) )
    {
      if ( !v28 )
      {
        sub_157F2D0(v18, *(_QWORD *)(v17 + 40), 0);
        sub_157F2D0(v20, *(_QWORD *)(v17 + 40), 0);
        goto LABEL_71;
      }
      v33 = *(_BYTE *)(v28 + 16);
      v30 = 0;
      if ( v33 != 77 )
        goto LABEL_47;
      goto LABEL_67;
    }
    v31 = *(_BYTE *)(v30 + 16);
    if ( v31 == 77 && v18 == *(_QWORD *)(v30 + 40) )
      break;
    if ( !v28 )
    {
      if ( v31 == 5 )
        goto LABEL_44;
      sub_157F2D0(v18, *(_QWORD *)(v17 + 40), 0);
      sub_157F2D0(v20, *(_QWORD *)(v17 + 40), 0);
LABEL_89:
      BUG();
    }
    if ( *(_BYTE *)(v28 + 16) == 77 )
      goto LABEL_67;
LABEL_43:
    if ( *(_BYTE *)(v30 + 16) != 5 )
      goto LABEL_46;
LABEL_44:
    v61 = v28;
    v32 = sub_1593DF0(v30, v24, v27, v25);
    v28 = v61;
    if ( !v32 )
      goto LABEL_45;
LABEL_25:
    v14 = v72;
    v15 = v71;
    if ( !(_DWORD)v72 )
      goto LABEL_26;
  }
  v24 = *(_QWORD *)(v17 + 40);
  v65 = v28;
  v56 = sub_1455EB0(v30, v24);
  v28 = v65;
  v30 = v56;
  if ( !v65 )
    goto LABEL_85;
  v33 = *(_BYTE *)(v65 + 16);
  if ( v33 != 77 )
  {
LABEL_68:
    if ( !v30 )
      goto LABEL_47;
    goto LABEL_43;
  }
LABEL_67:
  v33 = 77;
  if ( v20 != *(_QWORD *)(v28 + 40) )
    goto LABEL_68;
  v24 = *(_QWORD *)(v17 + 40);
  v28 = sub_1455EB0(v28, v24);
LABEL_85:
  if ( v30 && *(_BYTE *)(v30 + 16) == 5 )
    goto LABEL_44;
LABEL_45:
  if ( !v28 )
  {
    sub_157F2D0(v18, *(_QWORD *)(v17 + 40), 0);
    sub_157F2D0(v20, *(_QWORD *)(v17 + 40), 0);
    if ( !v30 )
      goto LABEL_71;
    goto LABEL_89;
  }
LABEL_46:
  v33 = *(_BYTE *)(v28 + 16);
LABEL_47:
  if ( v33 == 5 )
  {
    v62 = v28;
    v34 = sub_1593DF0(v28, v24, v27, v25);
    v28 = v62;
    if ( v34 )
      goto LABEL_25;
  }
  v63 = v28;
  sub_157F2D0(v18, *(_QWORD *)(v17 + 40), 0);
  sub_157F2D0(v20, *(_QWORD *)(v17 + 40), 0);
  v35 = v63;
  if ( !v30 )
    goto LABEL_71;
  if ( v30 == v63 )
    goto LABEL_54;
  if ( *(_BYTE *)(v63 + 16) == 9 )
  {
    v35 = v30;
LABEL_54:
    v64 = v35;
    v67 = 257;
    v37 = a3[3];
    v38 = sub_1648A60(56, 1u);
    v39 = (__int64)v38;
    if ( v38 )
      sub_15F6F90((__int64)v38, v37, v64, 0);
    goto LABEL_56;
  }
  if ( *(_BYTE *)(v30 + 16) == 9 )
    goto LABEL_54;
  v36 = *(_QWORD *)(v17 - 72);
  v66[0] = (__int64)"retval";
  v67 = 259;
  v35 = sub_156B790(a3, v36, v30, v63, (__int64)v66, v17);
  if ( v35 )
    goto LABEL_54;
LABEL_71:
  v67 = 257;
  v47 = a3[3];
  v48 = sub_1648A60(56, 0);
  v39 = (__int64)v48;
  if ( v48 )
    sub_15F6F90((__int64)v48, v47, 0, 0);
LABEL_56:
  v40 = a3[1];
  if ( v40 )
  {
    v41 = (__int64 *)a3[2];
    sub_157E9D0(v40 + 40, v39);
    v42 = *(_QWORD *)(v39 + 24);
    v43 = *v41;
    *(_QWORD *)(v39 + 32) = v41;
    v43 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v39 + 24) = v43 | v42 & 7;
    *(_QWORD *)(v43 + 8) = v39 + 24;
    *v41 = *v41 & 7 | (v39 + 24);
  }
  sub_164B780(v39, v66);
  v44 = v39;
LABEL_59:
  v3 = 1;
  sub_12A86E0(a3, v44);
  sub_1B44FE0(v17);
  v15 = v71;
LABEL_27:
  if ( v15 != v73 )
    _libc_free((unsigned __int64)v15);
LABEL_29:
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
  return v3;
}
