// Function: sub_1248770
// Address: 0x1248770
//
__int64 __fastcall sub_1248770(__int64 a1, __int64 *a2)
{
  unsigned __int64 v3; // r14
  int v4; // eax
  unsigned int v5; // edx
  unsigned __int64 *v6; // r14
  unsigned __int64 v7; // rsi
  __int64 v8; // r9
  int v9; // eax
  unsigned __int64 *v10; // rdx
  bool v11; // zf
  unsigned __int64 v12; // r13
  int v13; // eax
  unsigned int v14; // r15d
  int v15; // eax
  __int64 v16; // rax
  unsigned __int64 *v17; // r13
  unsigned __int64 *v18; // r15
  _QWORD *v19; // rax
  unsigned __int64 *v20; // r15
  unsigned __int64 *v21; // r13
  __int64 v22; // rdi
  unsigned __int64 *v23; // r12
  __int64 v24; // rax
  unsigned __int64 *v25; // rbx
  __int64 v26; // rdi
  __int64 v28; // rax
  __int64 v29; // r15
  __int64 v30; // r13
  __int64 v31; // rdi
  int v32; // eax
  _BYTE *v33; // rsi
  __int64 v34; // rcx
  unsigned __int64 *v35; // rdx
  unsigned __int64 *v36; // r15
  _QWORD *v37; // rax
  _QWORD *v38; // rcx
  unsigned __int64 *v39; // r13
  __int64 v40; // rdi
  int v41; // eax
  int v42; // eax
  int v43; // [rsp+8h] [rbp-138h]
  __int64 v44; // [rsp+10h] [rbp-130h]
  __int64 v45; // [rsp+18h] [rbp-128h]
  unsigned __int8 v46; // [rsp+38h] [rbp-108h]
  __int64 i; // [rsp+38h] [rbp-108h]
  __int64 v48; // [rsp+38h] [rbp-108h]
  unsigned int v49; // [rsp+40h] [rbp-100h]
  _QWORD *v51; // [rsp+50h] [rbp-F0h] BYREF
  unsigned __int64 v52; // [rsp+58h] [rbp-E8h] BYREF
  _QWORD v53[2]; // [rsp+60h] [rbp-E0h] BYREF
  _QWORD v54[2]; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD *v55; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v56; // [rsp+88h] [rbp-B8h]
  _QWORD v57[2]; // [rsp+90h] [rbp-B0h] BYREF
  unsigned __int64 v58[4]; // [rsp+A0h] [rbp-A0h] BYREF
  char v59; // [rsp+C0h] [rbp-80h]
  char v60; // [rsp+C1h] [rbp-7Fh]
  unsigned __int64 *v61; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v62; // [rsp+D8h] [rbp-68h]
  _BYTE v63[96]; // [rsp+E0h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(a1 + 232);
  v53[0] = v54;
  v4 = *(_DWORD *)(a1 + 240);
  v53[1] = 0;
  LOBYTE(v54[0]) = 0;
  if ( v4 == 507 )
  {
    sub_2240AE0(v53, a1 + 248);
    v32 = sub_1205200(a1 + 176);
    v5 = -1;
    *(_DWORD *)(a1 + 240) = v32;
  }
  else
  {
    v5 = -1;
    if ( v4 == 502 )
    {
      v49 = *(_DWORD *)(a1 + 280);
      v42 = sub_1205200(a1 + 176);
      v5 = v49;
      *(_DWORD *)(a1 + 240) = v42;
    }
  }
  v6 = (unsigned __int64 *)sub_1248090((__int64)a2, v53, v5, v3);
  if ( !v6 )
  {
    v46 = 1;
    goto LABEL_43;
  }
  LOBYTE(v57[0]) = 0;
  v55 = v57;
  v61 = (unsigned __int64 *)v63;
  v62 = 0x600000000LL;
  v56 = 0;
  while ( 1 )
  {
    while ( *(_DWORD *)(a1 + 240) == 17 )
    {
      while ( 1 )
      {
        if ( *(_BYTE *)(a1 + 1746) )
        {
          v7 = *(_QWORD *)(a1 + 232);
          v60 = 1;
          v58[0] = (unsigned __int64)"debug record should not appear in a module containing debug info intrinsics";
          v59 = 3;
          sub_11FD800(a1 + 176, v7, (__int64)v58, 1);
          goto LABEL_34;
        }
        if ( !*(_BYTE *)(a1 + 1745) )
        {
          v28 = *(_QWORD *)(a1 + 344);
          v29 = *(_QWORD *)(v28 + 32);
          v30 = v28 + 24;
          for ( i = v28; v30 != v29; v29 = *(_QWORD *)(v29 + 8) )
          {
            v31 = v29 - 56;
            if ( !v29 )
              v31 = 0;
            sub_B2BA20(v31, 1u);
          }
          *(_BYTE *)(i + 872) = 1;
        }
        *(_BYTE *)(a1 + 1745) = 1;
        v7 = (unsigned __int64)&v52;
        *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
        if ( (unsigned __int8)sub_122E950(a1, (__int64 *)&v52, a2) )
          goto LABEL_34;
        v9 = v62;
        if ( HIDWORD(v62) <= (unsigned int)v62 )
          break;
        v10 = &v61[(unsigned int)v62];
        if ( v10 )
        {
          *v10 = v52;
          v9 = v62;
        }
        v11 = *(_DWORD *)(a1 + 240) == 17;
        LODWORD(v62) = v9 + 1;
        if ( !v11 )
          goto LABEL_14;
      }
      v33 = v63;
      v48 = sub_C8D7D0((__int64)&v61, (__int64)v63, 0, 8u, v58, v8);
      v34 = (unsigned int)v62;
      if ( v34 * 8 + v48 )
      {
        *(_QWORD *)(v34 * 8 + v48) = v52;
        v34 = (unsigned int)v62;
      }
      v35 = v61;
      v36 = &v61[v34];
      if ( v61 != &v61[v34] )
      {
        v37 = (_QWORD *)v48;
        v38 = (_QWORD *)(v48 + v34 * 8);
        do
        {
          if ( v37 )
          {
            v33 = (_BYTE *)*v35;
            *v37 = *v35;
            *v35 = 0;
          }
          ++v37;
          ++v35;
        }
        while ( v37 != v38 );
        v39 = v61;
        v36 = &v61[(unsigned int)v62];
        if ( v61 != v36 )
        {
          do
          {
            v40 = *--v36;
            if ( v40 )
              sub_B12320(v40);
          }
          while ( v39 != v36 );
          v36 = v61;
        }
      }
      v41 = v58[0];
      if ( v36 != (unsigned __int64 *)v63 )
      {
        v43 = v58[0];
        _libc_free(v36, v33);
        v41 = v43;
      }
      LODWORD(v62) = v62 + 1;
      HIDWORD(v62) = v41;
      v61 = (unsigned __int64 *)v48;
    }
LABEL_14:
    v12 = *(_QWORD *)(a1 + 232);
    sub_2241130(&v55, 0, v56, byte_3F871B3, 0);
    v13 = *(_DWORD *)(a1 + 240);
    if ( v13 != 504 )
      break;
    v14 = *(_DWORD *)(a1 + 280);
    v7 = 3;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    if ( (unsigned __int8)sub_120AFE0(a1, 3, "expected '=' after instruction id") )
      goto LABEL_34;
LABEL_17:
    v7 = (unsigned __int64)&v51;
    v15 = sub_1237990(a1, (__int64 *)&v51, (__int64)v6, a2);
    if ( v15 == 1 )
      goto LABEL_34;
    if ( v15 == 2 )
    {
      v16 = v45;
      LOWORD(v16) = 0;
      v45 = v16;
      sub_B44240(v51, (__int64)v6, v6 + 6, v16);
    }
    else
    {
      if ( v15 )
        BUG();
      v24 = v44;
      LOWORD(v24) = 0;
      v44 = v24;
      sub_B44240(v51, (__int64)v6, v6 + 6, v24);
      if ( *(_DWORD *)(a1 + 240) != 4 )
        goto LABEL_21;
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    }
    v7 = (unsigned __int64)v51;
    if ( (unsigned __int8)sub_123FA00(a1, (__int64)v51) )
      goto LABEL_34;
LABEL_21:
    v7 = v14;
    v46 = sub_1247630(a2, v14, (__int64)&v55, v12, (__int64)v51);
    if ( v46 )
      goto LABEL_34;
    v17 = &v61[(unsigned int)v62];
    v18 = v61;
    if ( v17 != v61 )
    {
      do
      {
        v7 = *v18++;
        v19 = v51;
        *(v18 - 1) = 0;
        sub_AA8770((__int64)v6, v7, (__int64)(v19 + 3), 0);
      }
      while ( v17 != v18 );
      v20 = v61;
      v21 = &v61[(unsigned int)v62];
      while ( v20 != v21 )
      {
        while ( 1 )
        {
          v22 = *--v21;
          if ( !v22 )
            break;
          sub_B12320(v22);
          if ( v20 == v21 )
            goto LABEL_28;
        }
      }
    }
LABEL_28:
    LODWORD(v62) = 0;
    if ( (unsigned int)*(unsigned __int8 *)v51 - 30 <= 0xA )
    {
      v23 = v61;
      goto LABEL_39;
    }
  }
  if ( v13 != 510
    || (sub_2240AE0(&v55, a1 + 248),
        v7 = 3,
        *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176),
        !(unsigned __int8)sub_120AFE0(a1, 3, "expected '=' after instruction name")) )
  {
    v14 = -1;
    goto LABEL_17;
  }
LABEL_34:
  v25 = v61;
  v23 = &v61[(unsigned int)v62];
  if ( v61 == v23 )
  {
    v46 = 1;
  }
  else
  {
    do
    {
      v26 = *--v23;
      if ( v26 )
        sub_B12320(v26);
    }
    while ( v25 != v23 );
    v46 = 1;
    v23 = v61;
  }
LABEL_39:
  if ( v23 != (unsigned __int64 *)v63 )
    _libc_free(v23, v7);
  if ( v55 != v57 )
    j_j___libc_free_0(v55, v57[0] + 1LL);
LABEL_43:
  if ( (_QWORD *)v53[0] != v54 )
    j_j___libc_free_0(v53[0], v54[0] + 1LL);
  return v46;
}
