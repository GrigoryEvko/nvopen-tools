// Function: sub_EEA350
// Address: 0xeea350
//
__int64 __fastcall sub_EEA350(char **a1, char a2)
{
  char *v3; // rdx
  char *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r12
  bool v13; // zf
  char *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r15
  char *v21; // rax
  char *v22; // rdx
  char *v23; // rax
  int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  char v29; // al
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 *v34; // rsi
  _QWORD *v35; // rax
  __int64 *v36; // rax
  __int64 v37; // rax
  char v38; // r15
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  _QWORD *v43; // rax
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rax
  __int64 v53; // rsi
  __int64 v54; // rax
  char v55; // [rsp+8h] [rbp-E8h]
  __int64 v56; // [rsp+8h] [rbp-E8h]
  __int64 v57; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v58; // [rsp+20h] [rbp-D0h] BYREF
  __int64 *v59; // [rsp+28h] [rbp-C8h] BYREF
  _BYTE *v60; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v61; // [rsp+38h] [rbp-B8h]
  _BYTE v62[176]; // [rsp+40h] [rbp-B0h] BYREF

  v57 = 0;
  if ( (unsigned __int8)sub_EE3B50((const void **)a1, 3u, "srN") )
  {
    v57 = sub_EED260(a1);
    if ( !v57 )
      return 0;
    v3 = a1[1];
    v4 = *a1;
    if ( *a1 != v3 && *v4 == 73 )
    {
      v60 = (_BYTE *)sub_EEFA10(a1, 0);
      if ( !v60 )
        return 0;
      v57 = sub_EE7CC0((__int64)(a1 + 101), &v57, (unsigned __int64 *)&v60, v49, v50, v51);
      if ( !v57 )
        return 0;
      v4 = *a1;
      v3 = a1[1];
    }
    if ( v3 == v4 || *v4 != 69 )
    {
      do
      {
        v60 = (_BYTE *)sub_EF0590(a1);
        if ( !v60 )
          return 0;
        v57 = sub_EE8DA0((__int64)(a1 + 101), &v57, (unsigned __int64 *)&v60, v5, v6, v7);
        if ( !v57 )
          return 0;
        v4 = *a1;
        v3 = a1[1];
      }
      while ( v3 == *a1 || *v4 != 69 );
    }
    *a1 = v4 + 1;
    if ( v4 + 1 != v3 && (unsigned int)(v4[1] - 48) <= 9 )
      goto LABEL_13;
    goto LABEL_32;
  }
  v13 = (unsigned __int8)sub_EE3B50((const void **)a1, 2u, "sr") == 0;
  v14 = *a1;
  if ( !v13 )
  {
    if ( a1[1] == v14 || (unsigned int)(*v14 - 48) > 9 )
    {
      v57 = sub_EED260(a1);
      if ( !v57 )
        return 0;
      if ( a1[1] != *a1 )
      {
        v24 = **a1;
        if ( (_BYTE)v24 == 73 )
        {
          v60 = (_BYTE *)sub_EEFA10(a1, 0);
          if ( !v60 )
            return 0;
          v57 = sub_EE7CC0((__int64)(a1 + 101), &v57, (unsigned __int64 *)&v60, v46, v47, v48);
          if ( !v57 )
            return 0;
          v22 = a1[1];
          v23 = *a1;
LABEL_26:
          if ( v22 == v23 )
            goto LABEL_32;
          v24 = *v23;
        }
        if ( (unsigned int)(v24 - 48) <= 9 )
        {
LABEL_13:
          v60 = (_BYTE *)sub_EF0590(a1);
          if ( !v60 )
            return 0;
          return sub_EE8DA0((__int64)(a1 + 101), &v57, (unsigned __int64 *)&v60, v8, v9, v10);
        }
      }
LABEL_32:
      v60 = (_BYTE *)sub_EF4590(a1);
      if ( !v60 )
        return 0;
      return sub_EE8DA0((__int64)(a1 + 101), &v57, (unsigned __int64 *)&v60, v8, v9, v10);
    }
    while ( 1 )
    {
      v15 = sub_EF0590(a1);
      v58 = v15;
      if ( !v15 )
        return 0;
      if ( v57 )
        break;
      if ( a2 )
      {
        v56 = v15;
        v60 = v62;
        v38 = *((_BYTE *)a1 + 937);
        v61 = 0x2000000000LL;
        sub_D953B0((__int64)&v60, 46, v16, v17, (__int64)v62, v19);
        sub_D953B0((__int64)&v60, v56, v39, v40, v41, v42);
        v43 = sub_C65B40((__int64)(a1 + 113), (__int64)&v60, (__int64 *)&v59, (__int64)off_497B2F0);
        if ( v43 )
        {
          v20 = (__int64)(v43 + 1);
          if ( v60 != v62 )
            _libc_free(v60, &v60);
          v60 = (_BYTE *)v20;
          v44 = sub_EE6840((__int64)(a1 + 118), (__int64 *)&v60);
          if ( v44 )
          {
            v45 = v44[1];
            if ( v45 )
              v20 = v45;
          }
          if ( a1[116] == (char *)v20 )
            *((_BYTE *)a1 + 936) = 1;
        }
        else
        {
          if ( !v38 )
          {
            if ( v60 != v62 )
              _libc_free(v60, &v60);
            a1[115] = 0;
            return 0;
          }
          v52 = sub_CD1D40((__int64 *)a1 + 101, 32, 3);
          *(_QWORD *)v52 = 0;
          v53 = v52;
          v20 = v52 + 8;
          *(_WORD *)(v52 + 16) = 16430;
          *(_BYTE *)(v52 + 18) = *(_BYTE *)(v52 + 18) & 0xF0 | 5;
          *(_QWORD *)(v52 + 8) = &unk_49DFF08;
          *(_QWORD *)(v52 + 24) = v58;
          sub_C657C0((__int64 *)a1 + 113, (__int64 *)v52, v59, (__int64)off_497B2F0);
          if ( v60 != v62 )
            _libc_free(v60, v53);
          a1[115] = (char *)v20;
        }
        v57 = v20;
        goto LABEL_22;
      }
      v57 = v15;
LABEL_23:
      v21 = *a1;
      v22 = a1[1];
      if ( *a1 != v22 && *v21 == 69 )
      {
        v23 = v21 + 1;
        *a1 = v23;
        goto LABEL_26;
      }
    }
    v57 = sub_EE8DA0((__int64)(a1 + 101), &v57, (unsigned __int64 *)&v58, v17, v18, v19);
    v20 = v57;
LABEL_22:
    if ( !v20 )
      return 0;
    goto LABEL_23;
  }
  if ( a1[1] == v14 || (unsigned int)(*v14 - 48) > 9 )
    v11 = sub_EF4590(a1);
  else
    v11 = sub_EF0590(a1);
  v57 = v11;
  if ( !v11 )
    return 0;
  if ( a2 )
  {
    v29 = *((_BYTE *)a1 + 937);
    v60 = v62;
    v55 = v29;
    v61 = 0x2000000000LL;
    sub_D953B0((__int64)&v60, 46, v25, v26, v27, v28);
    sub_D953B0((__int64)&v60, v11, v30, v31, v32, v33);
    v34 = (__int64 *)&v60;
    v35 = sub_C65B40((__int64)(a1 + 113), (__int64)&v60, (__int64 *)&v59, (__int64)off_497B2F0);
    v11 = (__int64)v35;
    if ( v35 )
    {
      v11 = (__int64)(v35 + 1);
      if ( v60 != v62 )
        _libc_free(v60, &v60);
      v60 = (_BYTE *)v11;
      v36 = sub_EE6840((__int64)(a1 + 118), (__int64 *)&v60);
      if ( v36 )
      {
        v37 = v36[1];
        if ( v37 )
          v11 = v37;
      }
      if ( a1[116] == (char *)v11 )
        *((_BYTE *)a1 + 936) = 1;
    }
    else
    {
      if ( v55 )
      {
        v54 = sub_CD1D40((__int64 *)a1 + 101, 32, 3);
        *(_QWORD *)v54 = 0;
        v34 = (__int64 *)v54;
        v11 = v54 + 8;
        *(_WORD *)(v54 + 16) = 16430;
        *(_BYTE *)(v54 + 18) = *(_BYTE *)(v54 + 18) & 0xF0 | 5;
        *(_QWORD *)(v54 + 8) = &unk_49DFF08;
        *(_QWORD *)(v54 + 24) = v57;
        sub_C657C0((__int64 *)a1 + 113, (__int64 *)v54, v59, (__int64)off_497B2F0);
      }
      if ( v60 != v62 )
        _libc_free(v60, v34);
      a1[115] = (char *)v11;
    }
  }
  return v11;
}
