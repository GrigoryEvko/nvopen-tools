// Function: sub_12B9C40
// Address: 0x12b9c40
//
__int64 __fastcall sub_12B9C40(__int64 *a1, __int64 (*a2)())
{
  char v3; // r14
  __int64 v4; // r12
  unsigned int v5; // r12d
  _QWORD *v6; // r13
  _QWORD *v7; // r15
  __int64 v8; // rdi
  _QWORD *v9; // r13
  _QWORD *v10; // r15
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  _QWORD *v15; // r13
  _QWORD *v16; // r15
  __int64 v17; // rdi
  _QWORD *v18; // r13
  _QWORD *v19; // r15
  __int64 v20; // rdi
  __int64 v21; // rdi
  _QWORD *v23; // [rsp+0h] [rbp-40h]
  _QWORD *v24; // [rsp+0h] [rbp-40h]
  __int64 v25; // [rsp+8h] [rbp-38h]

  v3 = byte_4F92D70;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
    {
      a2 = sub_12B9A60;
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    }
    v25 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( !a1 || (v4 = *a1) == 0 )
    {
      v5 = 5;
LABEL_56:
      sub_16C30E0(v25);
      return v5;
    }
    v3 = 1;
  }
  else
  {
    if ( !qword_4F92D80 )
    {
      a2 = sub_12B9A60;
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    }
    v25 = qword_4F92D80;
    if ( !a1 )
      return 5;
    v4 = *a1;
    if ( !*a1 )
      return 5;
  }
  v6 = *(_QWORD **)(v4 + 8);
  v23 = *(_QWORD **)v4;
  if ( *(_QWORD **)v4 != v6 )
  {
    v7 = *(_QWORD **)v4;
    do
    {
      v8 = v7[2];
      if ( v8 )
        _libc_free(v8, a2);
      if ( *v7 )
        _libc_free(*v7, a2);
      v7 += 4;
    }
    while ( v6 != v7 );
    *(_QWORD *)(v4 + 8) = v23;
  }
  v9 = *(_QWORD **)(v4 + 32);
  v24 = *(_QWORD **)(v4 + 24);
  if ( v24 != v9 )
  {
    v10 = *(_QWORD **)(v4 + 24);
    do
    {
      v11 = v10[2];
      if ( v11 )
        _libc_free(v11, a2);
      if ( *v10 )
        _libc_free(*v10, a2);
      v10 += 4;
    }
    while ( v9 != v10 );
    *(_QWORD *)(v4 + 32) = v24;
  }
  v12 = *(_QWORD *)(v4 + 184);
  *(_QWORD *)(v4 + 112) = 0;
  *(_QWORD *)(v4 + 120) = 0;
  *(_QWORD *)(v4 + 128) = 0;
  *(_QWORD *)(v4 + 136) = 0;
  *(_QWORD *)(v4 + 144) = 0;
  *(_QWORD *)(v4 + 152) = 0;
  *(_QWORD *)(v4 + 160) = 0;
  *(_QWORD *)(v4 + 168) = 0;
  *(_DWORD *)(v4 + 176) = 0;
  *(_QWORD *)(v4 + 208) = 0;
  *(_QWORD *)(v4 + 216) = 0;
  if ( v12 )
  {
    a2 = (__int64 (*)())(*(_QWORD *)(v4 + 200) - v12);
    j_j___libc_free_0(v12, a2);
  }
  v13 = *(_QWORD *)(v4 + 80);
  if ( v13 != v4 + 96 )
  {
    a2 = (__int64 (*)())(*(_QWORD *)(v4 + 96) + 1LL);
    j_j___libc_free_0(v13, a2);
  }
  v14 = *(_QWORD *)(v4 + 48);
  if ( v14 != v4 + 64 )
  {
    a2 = (__int64 (*)())(*(_QWORD *)(v4 + 64) + 1LL);
    j_j___libc_free_0(v14, a2);
  }
  v15 = *(_QWORD **)(v4 + 32);
  v16 = *(_QWORD **)(v4 + 24);
  if ( v15 != v16 )
  {
    do
    {
      v17 = v16[2];
      if ( v17 )
        _libc_free(v17, a2);
      if ( *v16 )
        _libc_free(*v16, a2);
      v16 += 4;
    }
    while ( v15 != v16 );
    v16 = *(_QWORD **)(v4 + 24);
  }
  if ( v16 )
  {
    a2 = (__int64 (*)())(*(_QWORD *)(v4 + 40) - (_QWORD)v16);
    j_j___libc_free_0(v16, a2);
  }
  v18 = *(_QWORD **)(v4 + 8);
  v19 = *(_QWORD **)v4;
  if ( v18 != *(_QWORD **)v4 )
  {
    do
    {
      v20 = v19[2];
      if ( v20 )
        _libc_free(v20, a2);
      if ( *v19 )
        _libc_free(*v19, a2);
      v19 += 4;
    }
    while ( v18 != v19 );
    v19 = *(_QWORD **)v4;
  }
  if ( v19 )
    j_j___libc_free_0(v19, *(_QWORD *)(v4 + 16) - (_QWORD)v19);
  v21 = v4;
  v5 = 0;
  j_j___libc_free_0(v21, 224);
  *a1 = 0;
  if ( v3 )
    goto LABEL_56;
  return v5;
}
