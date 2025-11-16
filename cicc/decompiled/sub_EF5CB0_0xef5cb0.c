// Function: sub_EF5CB0
// Address: 0xef5cb0
//
__int64 __fastcall sub_EF5CB0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rax
  unsigned int v4; // r13d
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r12
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD **v20; // rsi
  _QWORD *v21; // rax
  __int64 v22; // rdi
  __int64 *v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  _QWORD *v29; // r13
  _BYTE *v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r10
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  _QWORD *v40; // rax
  unsigned __int64 v41; // rax
  __int64 v42; // r10
  __int64 v43; // rdx
  __int64 v44; // r13
  __int64 v45; // r9
  __int64 v46; // r9
  unsigned __int64 v47; // r10
  _QWORD *v48; // rax
  __int64 v49; // r13
  __int64 *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int16 v53; // dx
  __int64 v54; // r14
  __int16 v55; // cx
  __int64 v56; // rax
  char v57; // [rsp+Fh] [rbp-F1h]
  _QWORD *v58; // [rsp+10h] [rbp-F0h]
  char v59; // [rsp+10h] [rbp-F0h]
  _QWORD *v60; // [rsp+18h] [rbp-E8h]
  char v61; // [rsp+20h] [rbp-E0h]
  __int64 v62; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v63; // [rsp+28h] [rbp-D8h]
  __int64 v64; // [rsp+28h] [rbp-D8h]
  __int64 *v65; // [rsp+38h] [rbp-C8h] BYREF
  _QWORD *v66; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+48h] [rbp-B8h]
  _QWORD v68[22]; // [rsp+50h] [rbp-B0h] BYREF

  v3 = *(_BYTE **)a1;
  if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v3 != 85 )
  {
    v4 = sub_EE3340(a1);
    v13 = sub_EF1F20(a1, a2, v5, v6, v7, v8);
    if ( !v13 || !v4 )
      return v13;
    v15 = *(_BYTE *)(a1 + 937);
    v66 = v68;
    v61 = v15;
    v67 = 0x2000000002LL;
    v68[0] = 3;
    sub_D953B0((__int64)&v66, v13, v9, v10, v11, v12);
    sub_D953B0((__int64)&v66, v4, v16, v17, v18, v19);
    v20 = &v66;
    v21 = sub_C65B40(a1 + 904, (__int64)&v66, (__int64 *)&v65, (__int64)off_497B2F0);
    if ( v21 )
    {
      v22 = (__int64)v66;
      v13 = (__int64)(v21 + 1);
      if ( v66 == v68 )
      {
LABEL_9:
        v66 = (_QWORD *)v13;
        v23 = sub_EE6840(a1 + 944, (__int64 *)&v66);
        if ( v23 )
        {
          v24 = v23[1];
          if ( v24 )
            v13 = v24;
        }
LABEL_12:
        if ( *(_QWORD *)(a1 + 928) == v13 )
          *(_BYTE *)(a1 + 936) = 1;
        return v13;
      }
LABEL_8:
      _libc_free(v22, &v66);
      goto LABEL_9;
    }
    if ( v61 )
    {
      v52 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
      v53 = *(_WORD *)(v52 + 16);
      *(_QWORD *)v52 = 0;
      v20 = (_QWORD **)v52;
      v54 = v52 + 8;
      v55 = *(_WORD *)(v13 + 9) & 0xFC0;
      *(_WORD *)(v52 + 16) = v53 & 0xC000 | 3;
      LOWORD(v52) = *(_WORD *)(v52 + 17);
      v20[3] = (_QWORD *)v13;
      v13 = v54;
      *((_DWORD *)v20 + 5) = v4;
      *(_WORD *)((char *)v20 + 17) = v55 | v52 & 0xF03F;
      v20[1] = &unk_49DEE88;
      sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v20, v65, (__int64)off_497B2F0);
    }
    else
    {
      v13 = 0;
    }
    goto LABEL_16;
  }
  *(_QWORD *)a1 = v3 + 1;
  v25 = sub_EE3BB0(a1);
  v63 = (unsigned __int8 *)v26;
  v29 = (_QWORD *)v25;
  if ( !v25 )
    return 0;
  v30 = *(_BYTE **)a1;
  if ( v25 > 8 )
  {
    v27 = v26;
    if ( *(_QWORD *)v26 == 0x746F7270636A626FLL && *(_BYTE *)(v26 + 8) == 111 )
    {
      *(_QWORD *)a1 = v26 + 9;
      *(_QWORD *)(a1 + 8) = v26 + v25;
      v41 = sub_EE3BB0(a1);
      *(_QWORD *)(a1 + 8) = v42;
      v64 = v43;
      v44 = v41;
      *(_QWORD *)a1 = v45;
      if ( v41 )
      {
        v47 = sub_EF5CB0(a1);
        if ( v47 )
        {
          v60 = (_QWORD *)v47;
          v59 = *(_BYTE *)(a1 + 937);
          v67 = 0x2000000000LL;
          v66 = v68;
          sub_EE3E30((__int64)&v66, 0xBu, v47, v44, v64, v46);
          v20 = &v66;
          v48 = sub_C65B40(a1 + 904, (__int64)&v66, (__int64 *)&v65, (__int64)off_497B2F0);
          v13 = (__int64)v48;
          if ( v48 )
          {
            v49 = (__int64)(v48 + 1);
            if ( v66 != v68 )
              _libc_free(v66, &v66);
            v66 = (_QWORD *)v49;
            v13 = v49;
            v50 = sub_EE6840(a1 + 944, (__int64 *)&v66);
            if ( v50 )
            {
              v13 = v50[1];
              if ( !v13 )
                v13 = v49;
            }
            goto LABEL_12;
          }
          if ( v59 )
          {
            v56 = sub_CD1D40((__int64 *)(a1 + 808), 48, 3);
            *(_QWORD *)v56 = 0;
            v20 = (_QWORD **)v56;
            v13 = v56 + 8;
            *(_WORD *)(v56 + 16) = 16395;
            LOBYTE(v56) = *(_BYTE *)(v56 + 18);
            v20[3] = v60;
            v20[4] = (_QWORD *)v44;
            *((_BYTE *)v20 + 18) = v56 & 0xF0 | 5;
            v20[1] = &unk_49DF1E8;
            v20[5] = (_QWORD *)v64;
            sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v20, v65, (__int64)off_497B2F0);
          }
          goto LABEL_16;
        }
      }
      return 0;
    }
  }
  v62 = 0;
  if ( *(_BYTE **)(a1 + 8) != v30 && *v30 == 73 )
  {
    v62 = sub_EEFA10(a1, 0, v26, v27, v28, (__int64)v30);
    if ( !v62 )
      return 0;
  }
  v35 = sub_EF5CB0(a1);
  if ( !v35 )
    return 0;
  v58 = (_QWORD *)v35;
  v57 = *(_BYTE *)(a1 + 937);
  v67 = 0x2000000002LL;
  v66 = v68;
  v68[0] = 2;
  sub_D953B0((__int64)&v66, v35, v31, v32, v33, v34);
  sub_C653C0((__int64)&v66, v63, (unsigned int)v29);
  sub_D953B0((__int64)&v66, v62, v36, v37, v38, v39);
  v20 = &v66;
  v40 = sub_C65B40(a1 + 904, (__int64)&v66, (__int64 *)&v65, (__int64)off_497B2F0);
  v13 = (__int64)v40;
  if ( v40 )
  {
    v22 = (__int64)v66;
    v13 = (__int64)(v40 + 1);
    if ( v66 == v68 )
      goto LABEL_9;
    goto LABEL_8;
  }
  if ( v57 )
  {
    v51 = sub_CD1D40((__int64 *)(a1 + 808), 56, 3);
    *(_QWORD *)v51 = 0;
    v20 = (_QWORD **)v51;
    v13 = v51 + 8;
    *(_WORD *)(v51 + 16) = 16386;
    LOBYTE(v51) = *(_BYTE *)(v51 + 18);
    v20[3] = v58;
    v20[4] = v29;
    *((_BYTE *)v20 + 18) = v51 & 0xF0 | 5;
    v20[1] = &unk_49DEE28;
    v20[5] = v63;
    v20[6] = (_QWORD *)v62;
    sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v20, v65, (__int64)off_497B2F0);
  }
LABEL_16:
  if ( v66 != v68 )
    _libc_free(v66, v20);
  *(_QWORD *)(a1 + 920) = v13;
  return v13;
}
