// Function: sub_1AEB5C0
// Address: 0x1aeb5c0
//
__int64 __fastcall sub_1AEB5C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        void (__fastcall *a5)(__int64 *, __int64, __int64),
        __int64 a6)
{
  __int64 v8; // rbx
  _QWORD *v9; // rax
  bool v10; // cc
  unsigned __int64 v11; // r8
  _BYTE *v12; // rdx
  __int64 *v13; // rbx
  _QWORD *v14; // r12
  __int64 v15; // r13
  __int64 *v16; // r12
  unsigned __int8 v17; // r14
  _QWORD *v18; // rax
  __int64 v19; // rsi
  __int64 *v20; // rax
  __int64 v21; // r9
  unsigned __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 *v25; // rcx
  __int64 v26; // rdi
  unsigned __int64 v27; // rsi
  __int64 v28; // rsi
  char v29; // al
  unsigned __int64 v30; // rdi
  _BYTE *v31; // rdx
  _BYTE *v32; // r15
  _QWORD **v33; // rax
  _QWORD *v34; // r13
  _QWORD **v35; // r12
  __int64 *v37; // r14
  __int64 v38; // r15
  __int64 *v39; // rbx
  _QWORD *v40; // r13
  _QWORD *v41; // rax
  _QWORD **v42; // rax
  _QWORD *v43; // rdi
  _QWORD *v44; // rsi
  _QWORD *v45; // rdx
  __int64 v47; // [rsp+8h] [rbp-C8h]
  unsigned __int8 v51; // [rsp+37h] [rbp-99h]
  __int64 *v52; // [rsp+38h] [rbp-98h]
  __int64 v53; // [rsp+38h] [rbp-98h]
  __int64 v54; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int8 v55; // [rsp+48h] [rbp-88h]
  __int64 *v56; // [rsp+50h] [rbp-80h] BYREF
  __int64 v57; // [rsp+58h] [rbp-78h]
  _BYTE v58[16]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v59; // [rsp+70h] [rbp-60h] BYREF
  _BYTE *v60; // [rsp+78h] [rbp-58h]
  _BYTE *v61; // [rsp+80h] [rbp-50h]
  __int64 v62; // [rsp+88h] [rbp-48h]
  int v63; // [rsp+90h] [rbp-40h]
  _BYTE v64[56]; // [rsp+98h] [rbp-38h] BYREF

  v8 = a1;
  v56 = (__int64 *)v58;
  v57 = 0x100000000LL;
  sub_1AEA440((__int64)&v56, a1);
  v51 = 0;
  if ( !(_DWORD)v57 )
    goto LABEL_47;
  v9 = v64;
  v10 = *(_BYTE *)(a2 + 16) <= 0x17u;
  v59 = 0;
  v60 = v64;
  v61 = v64;
  v62 = 1;
  v63 = 0;
  if ( v10 )
  {
    v11 = (unsigned __int64)v56;
    v12 = v64;
    v52 = &v56[(unsigned int)v57];
    goto LABEL_4;
  }
  v53 = sub_15F3430(a1);
  v37 = &v56[(unsigned int)v57];
  if ( v56 == v37 )
    goto LABEL_38;
  v38 = a4;
  v39 = v56;
  do
  {
    while ( 1 )
    {
      v40 = (_QWORD *)*v39;
      if ( a3 == v53 && a3 == sub_15F3430(*v39) )
      {
        sub_15F2300(v40, a3);
        v51 = 1;
        goto LABEL_52;
      }
      if ( !sub_15CCEE0(v38, a3, (__int64)v40) )
        break;
LABEL_52:
      if ( v37 == ++v39 )
        goto LABEL_57;
    }
    v41 = v60;
    if ( v61 == v60 )
    {
      v43 = &v60[8 * HIDWORD(v62)];
      if ( v60 != (_BYTE *)v43 )
      {
        v44 = 0;
        while ( v40 != (_QWORD *)*v41 )
        {
          if ( *v41 == -2 )
            v44 = v41;
          if ( v43 == ++v41 )
          {
            if ( !v44 )
              goto LABEL_82;
            *v44 = v40;
            --v63;
            ++v59;
            goto LABEL_52;
          }
        }
        goto LABEL_52;
      }
LABEL_82:
      if ( HIDWORD(v62) < (unsigned int)v62 )
      {
        ++HIDWORD(v62);
        *v43 = v40;
        ++v59;
        goto LABEL_52;
      }
    }
    ++v39;
    sub_16CCBA0((__int64)&v59, (__int64)v40);
  }
  while ( v37 != v39 );
LABEL_57:
  v11 = (unsigned __int64)v56;
  v8 = a1;
  v52 = &v56[(unsigned int)v57];
  if ( v52 != v56 )
  {
    v12 = v61;
    v9 = v60;
LABEL_4:
    v47 = v8;
    v13 = (__int64 *)v11;
    while ( 2 )
    {
      v15 = *v13;
      if ( v9 == (_QWORD *)v12 )
      {
        v14 = &v9[HIDWORD(v62)];
        if ( v14 == v9 )
        {
          v45 = v9;
        }
        else
        {
          do
          {
            if ( v15 == *v9 )
              break;
            ++v9;
          }
          while ( v14 != v9 );
          v45 = v14;
        }
LABEL_19:
        while ( v45 != v9 )
        {
          if ( *v9 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_8;
          ++v9;
        }
        if ( v14 == v9 )
          goto LABEL_21;
        goto LABEL_9;
      }
      v14 = &v12[8 * (unsigned int)v62];
      v9 = sub_16CC9F0((__int64)&v59, *v13);
      if ( v15 == *v9 )
      {
        if ( v61 == v60 )
          v45 = &v61[8 * HIDWORD(v62)];
        else
          v45 = &v61[8 * (unsigned int)v62];
        goto LABEL_19;
      }
      if ( v61 == v60 )
      {
        v9 = &v61[8 * HIDWORD(v62)];
        v45 = v9;
        goto LABEL_19;
      }
      v9 = &v61[8 * (unsigned int)v62];
LABEL_8:
      if ( v14 != v9 )
        goto LABEL_9;
LABEL_21:
      v16 = (__int64 *)sub_16498A0(v15);
      a5(&v54, a6, v15);
      v17 = v55;
      if ( v55 )
      {
        v18 = sub_1624210(a2);
        v19 = sub_1628DA0(v16, (__int64)v18);
        v20 = (__int64 *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
        if ( *v20 )
        {
          v21 = v20[1];
          v22 = v20[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v22 = v21;
          if ( v21 )
            *(_QWORD *)(v21 + 16) = *(_QWORD *)(v21 + 16) & 3LL | v22;
        }
        *v20 = v19;
        if ( v19 )
        {
          v23 = *(_QWORD *)(v19 + 8);
          v20[1] = v23;
          if ( v23 )
            *(_QWORD *)(v23 + 16) = (unsigned __int64)(v20 + 1) | *(_QWORD *)(v23 + 16) & 3LL;
          v20[2] = (v19 + 8) | v20[2] & 3;
          *(_QWORD *)(v19 + 8) = v20;
        }
        v24 = sub_1628DA0(v16, v54);
        v25 = (__int64 *)(v15 + 24 * (2LL - (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)));
        if ( *v25 )
        {
          v26 = v25[1];
          v27 = v25[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v27 = v26;
          if ( v26 )
            *(_QWORD *)(v26 + 16) = *(_QWORD *)(v26 + 16) & 3LL | v27;
        }
        *v25 = v24;
        if ( v24 )
        {
          v28 = *(_QWORD *)(v24 + 8);
          v25[1] = v28;
          if ( v28 )
            *(_QWORD *)(v28 + 16) = (unsigned __int64)(v25 + 1) | *(_QWORD *)(v28 + 16) & 3LL;
          v25[2] = (v24 + 8) | v25[2] & 3;
          *(_QWORD *)(v24 + 8) = v25;
        }
        v51 = v17;
        if ( v52 == ++v13 )
        {
LABEL_37:
          v8 = v47;
          break;
        }
      }
      else
      {
LABEL_9:
        if ( v52 == ++v13 )
          goto LABEL_37;
      }
      v12 = v61;
      v9 = v60;
      continue;
    }
  }
LABEL_38:
  if ( HIDWORD(v62) == v63 )
  {
LABEL_67:
    v30 = (unsigned __int64)v61;
    v31 = v60;
  }
  else
  {
    v29 = sub_1AEAA40(v8);
    v30 = (unsigned __int64)v61;
    v31 = v60;
    v51 |= v29;
    if ( v61 == v60 )
      v32 = &v61[8 * HIDWORD(v62)];
    else
      v32 = &v61[8 * (unsigned int)v62];
    if ( v61 != v32 )
    {
      v33 = (_QWORD **)v61;
      while ( 1 )
      {
        v34 = *v33;
        v35 = v33;
        if ( (unsigned __int64)*v33 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v32 == (_BYTE *)++v33 )
          goto LABEL_45;
      }
      if ( v32 != (_BYTE *)v33 )
      {
        do
        {
          if ( v8 == sub_1601A30((__int64)v34, 1) )
          {
            sub_15F20C0(v34);
            v51 = 1;
          }
          v42 = v35 + 1;
          if ( v35 + 1 == (_QWORD **)v32 )
            break;
          while ( 1 )
          {
            v34 = *v42;
            v35 = v42;
            if ( (unsigned __int64)*v42 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v32 == (_BYTE *)++v42 )
              goto LABEL_67;
          }
        }
        while ( v32 != (_BYTE *)v42 );
        goto LABEL_67;
      }
    }
  }
LABEL_45:
  if ( (_BYTE *)v30 != v31 )
    _libc_free(v30);
LABEL_47:
  if ( v56 != (__int64 *)v58 )
    _libc_free((unsigned __int64)v56);
  return v51;
}
