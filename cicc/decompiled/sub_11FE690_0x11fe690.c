// Function: sub_11FE690
// Address: 0x11fe690
//
__int64 __fastcall sub_11FE690(__int64 a1)
{
  __int64 v2; // r12
  int v3; // edi
  __int64 v4; // rdx
  unsigned __int8 *v5; // r14
  int v6; // edi
  unsigned __int8 *v7; // r13
  _DWORD *v9; // r14
  _DWORD *v10; // rax
  void *v11; // r13
  void *v12; // r12
  _DWORD *v13; // r14
  _DWORD *v14; // rax
  void *v15; // r13
  __int64 v16; // rax
  void **v17; // rbx
  _DWORD *v18; // r14
  _DWORD *v19; // rax
  void *v20; // r13
  __int64 v21; // rax
  _DWORD *v22; // r14
  _DWORD *v23; // rax
  void *v24; // r13
  __int64 v25; // rax
  _DWORD *v26; // r14
  _DWORD *v27; // rax
  void *v28; // r13
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  char v32; // [rsp+Fh] [rbp-71h]
  __int64 v33[2]; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int64 v34; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v35; // [rsp+28h] [rbp-58h]
  void *v36; // [rsp+30h] [rbp-50h] BYREF
  void **v37; // [rsp+38h] [rbp-48h]

  v2 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)a1 = v2 + 2;
  v3 = *(unsigned __int8 *)(v2 + 2);
  if ( (unsigned __int8)(v3 - 72) <= 0xAu )
  {
    v4 = 1081;
    if ( _bittest64(&v4, (unsigned int)(v3 - 72)) )
    {
      *(_QWORD *)a1 = v2 + 3;
      v5 = (unsigned __int8 *)(v2 + 4);
      v32 = *(_BYTE *)(v2 + 2);
      if ( isxdigit(*(unsigned __int8 *)(v2 + 3)) )
        goto LABEL_4;
LABEL_8:
      *(_QWORD *)a1 = v2 + 1;
      return 1;
    }
  }
  v32 = 74;
  v5 = (unsigned __int8 *)(v2 + 3);
  if ( !isxdigit(v3) )
    goto LABEL_8;
  do
  {
LABEL_4:
    *(_QWORD *)a1 = v5;
    v6 = *v5;
    v7 = v5++;
  }
  while ( isxdigit(v6) );
  if ( v32 != 74 )
  {
    switch ( v32 )
    {
      case 'H':
        v35 = 16;
        v34 = sub_11FE380(a1, (unsigned __int8 *)(v2 + 3), v7);
        v22 = sub_C332F0();
        v23 = sub_C33340();
        v24 = v23;
        if ( v22 == v23 )
          sub_C3C640(&v36, (__int64)v23, &v34);
        else
          sub_C3B160((__int64)&v36, v22, (__int64 *)&v34);
        sub_11FD130((void **)(a1 + 120), &v36);
        if ( v36 != v24 )
          goto LABEL_12;
        if ( !v37 )
          goto LABEL_13;
        v25 = 3LL * (_QWORD)*(v37 - 1);
        v17 = &v37[v25];
        while ( v37 != v17 )
        {
          v17 -= 3;
          if ( v24 == *v17 )
            sub_969EE0((__int64)v17);
          else
            sub_C338F0((__int64)v17);
        }
        break;
      case 'K':
        sub_11FE4C0(a1, v2 + 3, (__int64)v7, v33);
        sub_C438C0((__int64)&v34, 80, v33, 2u);
        v18 = sub_C33420();
        v19 = sub_C33340();
        v20 = v19;
        if ( v18 == v19 )
          sub_C3C640(&v36, (__int64)v19, &v34);
        else
          sub_C3B160((__int64)&v36, v18, (__int64 *)&v34);
        sub_11FD130((void **)(a1 + 120), &v36);
        if ( v36 != v20 )
          goto LABEL_12;
        if ( !v37 )
          goto LABEL_13;
        v21 = 3LL * (_QWORD)*(v37 - 1);
        v17 = &v37[v21];
        while ( v37 != v17 )
        {
          v17 -= 3;
          if ( v20 == *v17 )
            sub_969EE0((__int64)v17);
          else
            sub_C338F0((__int64)v17);
        }
        break;
      case 'L':
        sub_11FE5A0(a1, v2 + 3, (__int64)v7, v33);
        sub_C438C0((__int64)&v34, 128, v33, 2u);
        v13 = sub_C33330();
        v14 = sub_C33340();
        v15 = v14;
        if ( v13 == v14 )
          sub_C3C640(&v36, (__int64)v14, &v34);
        else
          sub_C3B160((__int64)&v36, v13, (__int64 *)&v34);
        sub_11FD130((void **)(a1 + 120), &v36);
        if ( v36 != v15 )
          goto LABEL_12;
        if ( !v37 )
          goto LABEL_13;
        v16 = 3LL * (_QWORD)*(v37 - 1);
        v17 = &v37[v16];
        while ( v37 != v17 )
        {
          v17 -= 3;
          if ( v15 == *v17 )
            sub_969EE0((__int64)v17);
          else
            sub_C338F0((__int64)v17);
        }
        break;
      case 'M':
        sub_11FE5A0(a1, v2 + 3, (__int64)v7, v33);
        sub_C438C0((__int64)&v34, 128, v33, 2u);
        v12 = sub_C33340();
        sub_C3C640(&v36, (__int64)v12, &v34);
        sub_11FD130((void **)(a1 + 120), &v36);
        if ( v12 != v36 )
          goto LABEL_12;
        if ( !v37 )
          goto LABEL_13;
        v31 = 3LL * (_QWORD)*(v37 - 1);
        v17 = &v37[v31];
        while ( v37 != v17 )
        {
          v17 -= 3;
          if ( v12 == *v17 )
            sub_969EE0((__int64)v17);
          else
            sub_C338F0((__int64)v17);
        }
        break;
      case 'R':
        v35 = 16;
        v34 = sub_11FE380(a1, (unsigned __int8 *)(v2 + 3), v7);
        v9 = sub_C33300();
        v10 = sub_C33340();
        v11 = v10;
        if ( v9 == v10 )
          sub_C3C640(&v36, (__int64)v10, &v34);
        else
          sub_C3B160((__int64)&v36, v9, (__int64 *)&v34);
        sub_11FD130((void **)(a1 + 120), &v36);
        if ( v36 != v11 )
          goto LABEL_12;
        if ( !v37 )
          goto LABEL_13;
        v30 = 3LL * (_QWORD)*(v37 - 1);
        v17 = &v37[v30];
        while ( v37 != v17 )
        {
          v17 -= 3;
          if ( v11 == *v17 )
            sub_969EE0((__int64)v17);
          else
            sub_C338F0((__int64)v17);
        }
        break;
      default:
        BUG();
    }
LABEL_49:
    j_j_j___libc_free_0_0(v17 - 1);
    goto LABEL_13;
  }
  v35 = 64;
  v34 = sub_11FE380(a1, (unsigned __int8 *)(v2 + 2), v7);
  v26 = sub_C33320();
  v27 = sub_C33340();
  v28 = v27;
  if ( v26 == v27 )
    sub_C3C640(&v36, (__int64)v27, &v34);
  else
    sub_C3B160((__int64)&v36, v26, (__int64 *)&v34);
  sub_11FD130((void **)(a1 + 120), &v36);
  if ( v36 == v28 )
  {
    if ( !v37 )
      goto LABEL_13;
    v29 = 3LL * (_QWORD)*(v37 - 1);
    v17 = &v37[v29];
    while ( v37 != v17 )
    {
      v17 -= 3;
      if ( v28 == *v17 )
        sub_969EE0((__int64)v17);
      else
        sub_C338F0((__int64)v17);
    }
    goto LABEL_49;
  }
LABEL_12:
  sub_C338F0((__int64)&v36);
LABEL_13:
  if ( v35 > 0x40 )
  {
    if ( v34 )
      j_j___libc_free_0_0(v34);
  }
  return 528;
}
