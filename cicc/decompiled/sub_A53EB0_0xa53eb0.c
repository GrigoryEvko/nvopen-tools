// Function: sub_A53EB0
// Address: 0xa53eb0
//
__int64 __fastcall sub_A53EB0(__int64 a1, __int64 *a2)
{
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rcx
  unsigned __int8 v13; // si
  unsigned __int8 v14; // si
  _BYTE *v15; // rax
  __int64 result; // rax
  __int64 v17; // rsi
  _QWORD **v18; // rdi
  char v19; // r12
  _BYTE *v20; // rax
  _QWORD **v21; // rax
  _BOOL4 v22; // r12d
  _QWORD *v23; // r12
  _BYTE *v24; // rsi
  __int64 v25; // rsi
  _QWORD *v26; // rdi
  _BYTE *v27; // rax
  _QWORD *v28; // rax
  _BYTE *v29; // rax
  _QWORD *v30; // [rsp+8h] [rbp-148h]
  double v31; // [rsp+18h] [rbp-138h]
  double v32; // [rsp+20h] [rbp-130h]
  __int64 v33; // [rsp+30h] [rbp-120h]
  __int64 v34; // [rsp+38h] [rbp-118h]
  char v35; // [rsp+4Fh] [rbp-101h] BYREF
  _QWORD *v36; // [rsp+50h] [rbp-100h] BYREF
  unsigned int v37; // [rsp+58h] [rbp-F8h]
  _QWORD *v38; // [rsp+60h] [rbp-F0h] BYREF
  _QWORD *v39; // [rsp+68h] [rbp-E8h]
  _BYTE *v40; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v41; // [rsp+88h] [rbp-C8h]
  __int64 v42; // [rsp+90h] [rbp-C0h]
  _BYTE v43[184]; // [rsp+98h] [rbp-B8h] BYREF

  v4 = *a2;
  v33 = *a2;
  v5 = sub_C33310(a1, a2);
  v34 = sub_C33320(a1);
  v9 = sub_C33340(a1, a2, v6, v7, v8);
  if ( v5 != v4 && v4 != v34 )
  {
    sub_904010(a1, "0x");
    if ( *a2 == v9 )
      sub_C3E660(&v36, a2);
    else
      sub_C3A850(&v36, a2);
    v10 = *a2;
    if ( v10 == sub_C33420() )
    {
      sub_A51310(a1, 0x4Bu);
      sub_C48300(&v38, &v36, 16);
      v29 = v38;
      if ( (unsigned int)v39 > 0x40 )
        v29 = (_BYTE *)*v38;
      v40 = v29;
      WORD2(v42) = 257;
      v41 = 0;
      LODWORD(v42) = 4;
      BYTE6(v42) = 0;
      sub_CB6AF0(a1, &v40);
      if ( (unsigned int)v39 > 0x40 && v38 )
        j_j___libc_free_0_0(v38);
      sub_C443A0(&v38, &v36, 64);
      v28 = v38;
      if ( (unsigned int)v39 <= 0x40 )
        goto LABEL_86;
    }
    else
    {
      v13 = 76;
      if ( v10 != sub_C33330() )
      {
        if ( v10 != v9 )
        {
          if ( v10 == sub_C332F0(&v36, 76, v11, v12) )
          {
            v14 = 72;
          }
          else
          {
            if ( v10 != sub_C33300() )
              BUG();
            v14 = 82;
          }
          sub_A51310(a1, v14);
          v15 = v36;
          if ( v37 > 0x40 )
            v15 = (_BYTE *)*v36;
          v40 = v15;
          v41 = 0;
          LODWORD(v42) = 4;
          WORD2(v42) = 257;
          BYTE6(v42) = 0;
          result = sub_CB6AF0(a1, &v40);
LABEL_14:
          if ( v37 > 0x40 )
          {
            if ( v36 )
              return j_j___libc_free_0_0(v36);
          }
          return result;
        }
        v13 = 77;
      }
      sub_A51310(a1, v13);
      sub_C443A0(&v38, &v36, 64);
      v27 = v38;
      if ( (unsigned int)v39 > 0x40 )
        v27 = (_BYTE *)*v38;
      v40 = v27;
      WORD2(v42) = 257;
      v41 = 0;
      LODWORD(v42) = 16;
      BYTE6(v42) = 0;
      sub_CB6AF0(a1, &v40);
      if ( (unsigned int)v39 > 0x40 && v38 )
        j_j___libc_free_0_0(v38);
      sub_C48300(&v38, &v36, 64);
      v28 = v38;
      if ( (unsigned int)v39 <= 0x40 )
      {
LABEL_86:
        v40 = v28;
        v41 = 0;
        LODWORD(v42) = 16;
        WORD2(v42) = 257;
        BYTE6(v42) = 0;
        result = sub_CB6AF0(a1, &v40);
        if ( (unsigned int)v39 > 0x40 && v38 )
          result = j_j___libc_free_0_0(v38);
        goto LABEL_14;
      }
    }
    v28 = (_QWORD *)*v28;
    goto LABEL_86;
  }
  if ( v33 == v9 )
  {
    if ( (*(_BYTE *)(a2[1] + 20) & 7u) <= 1 )
      goto LABEL_42;
  }
  else if ( (*((_BYTE *)a2 + 20) & 7u) < 2 )
  {
    goto LABEL_27;
  }
  v41 = 0;
  v40 = v43;
  v42 = 128;
  v31 = sub_C41B00(a2);
  if ( *a2 == v9 )
    sub_C40650(a2, &v40, 6, 0, 0);
  else
    sub_C35AD0(a2, &v40, 6, 0, 0);
  v17 = v34;
  sub_C43310(&v38, v34, v40, v41);
  v32 = sub_C41B00(&v38);
  if ( v38 == (_QWORD *)v9 )
  {
    if ( v39 )
    {
      v25 = 3LL * *(v39 - 1);
      v26 = &v39[v25];
      if ( v39 != &v39[v25] )
      {
        do
        {
          v30 = v26;
          v26 -= 3;
          if ( *v26 == v9 )
            sub_969EE0((__int64)v26);
          else
            sub_C338F0(v26);
        }
        while ( v39 != v26 );
        v25 = 3LL * *(v30 - 4);
      }
      v17 = v25 * 8 + 8;
      j_j_j___libc_free_0_0(v26 - 1);
    }
  }
  else
  {
    sub_C338F0(&v38);
  }
  if ( v32 == v31 )
  {
    v24 = v40;
    result = sub_CB6200(a1, v40, v41);
    if ( v40 != v43 )
      return _libc_free(v40, v24);
    return result;
  }
  if ( v40 != v43 )
    _libc_free(v40, v17);
  if ( v9 != *a2 )
  {
LABEL_27:
    sub_C33EB0(&v38, a2);
    if ( v33 != v34 )
      goto LABEL_28;
    goto LABEL_31;
  }
LABEL_42:
  sub_C3C790(&v38, a2);
  if ( v33 != v34 )
  {
LABEL_28:
    v18 = &v38;
    if ( v38 == (_QWORD *)v9 )
      v18 = (_QWORD **)v39;
    v19 = sub_C35FD0(v18);
    sub_C41640(&v38, v34, 1, &v35);
    if ( !v19 )
      goto LABEL_31;
    if ( v38 == (_QWORD *)v9 )
      sub_C3E660(&v36, &v38);
    else
      sub_C3A850(&v36, &v38);
    v21 = &v38;
    if ( v38 == (_QWORD *)v9 )
      v21 = (_QWORD **)v39;
    v22 = (*((_BYTE *)v21 + 20) & 8) != 0;
    if ( v9 == v34 )
      sub_C3C500(&v40, v9, 0);
    else
      sub_C373C0(&v40, v34, 0);
    if ( v40 == (_BYTE *)v9 )
      sub_C3D480(&v40, 1, v22, &v36);
    else
      sub_C36070(&v40, 1, v22, &v36);
    if ( v38 == (_QWORD *)v9 )
    {
      if ( v40 == (_BYTE *)v9 )
      {
        sub_969EE0((__int64)&v38);
        sub_C3C840(&v38, &v40);
        goto LABEL_55;
      }
    }
    else if ( v40 != (_BYTE *)v9 )
    {
      sub_C33870(&v38, &v40);
      goto LABEL_55;
    }
    sub_91D830(&v38);
    if ( v40 == (_BYTE *)v9 )
      sub_C3C840(&v38, &v40);
    else
      sub_C338E0(&v38, &v40);
LABEL_55:
    sub_91D830(&v40);
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
  }
LABEL_31:
  if ( v38 == (_QWORD *)v9 )
    sub_C3E660(&v36, &v38);
  else
    sub_C3A850(&v36, &v38);
  v20 = v36;
  if ( v37 > 0x40 )
    v20 = (_BYTE *)*v36;
  v40 = v20;
  WORD2(v42) = 257;
  v41 = 0;
  LODWORD(v42) = 0;
  BYTE6(v42) = 1;
  sub_CB6AF0(a1, &v40);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  if ( (_QWORD *)v9 != v38 )
    return sub_C338F0(&v38);
  result = (__int64)v39;
  if ( v39 )
  {
    v23 = &v39[3 * *(v39 - 1)];
    while ( v39 != v23 )
    {
      v23 -= 3;
      if ( v9 == *v23 )
        sub_969EE0((__int64)v23);
      else
        sub_C338F0(v23);
    }
    return j_j_j___libc_free_0_0(v23 - 1);
  }
  return result;
}
