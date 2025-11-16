// Function: sub_1B901A0
// Address: 0x1b901a0
//
__int64 __fastcall sub_1B901A0(__int64 a1)
{
  _BYTE *v2; // rdi
  int v3; // eax
  _BYTE *v4; // rdx
  _BYTE *v5; // r14
  _QWORD *v6; // rax
  __int64 v7; // r13
  _QWORD *v8; // rbx
  __int64 v9; // rax
  _QWORD *v10; // rbx
  _QWORD *v11; // r13
  unsigned __int64 v12; // rdi
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // r13
  _QWORD *v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // [rsp+0h] [rbp-70h] BYREF
  _BYTE *v20; // [rsp+8h] [rbp-68h]
  _BYTE *v21; // [rsp+10h] [rbp-60h]
  __int64 v22; // [rsp+18h] [rbp-58h]
  int v23; // [rsp+20h] [rbp-50h]
  _BYTE v24[72]; // [rsp+28h] [rbp-48h] BYREF

  v2 = v24;
  v3 = *(_DWORD *)(a1 + 64);
  v19 = 0;
  v20 = v24;
  v21 = v24;
  v22 = 4;
  v23 = 0;
  if ( !v3 )
    goto LABEL_2;
  v15 = *(_QWORD **)(a1 + 56);
  v16 = &v15[2 * *(unsigned int *)(a1 + 72)];
  if ( v15 == v16 )
    goto LABEL_2;
  while ( 1 )
  {
    v17 = v15;
    if ( *v15 != -16 && *v15 != -8 )
      break;
    v15 += 2;
    if ( v16 == v15 )
      goto LABEL_2;
  }
  if ( v15 == v16 )
  {
LABEL_2:
    v4 = v24;
    goto LABEL_3;
  }
  do
  {
    v18 = v17[1];
    v17 += 2;
    sub_1412190((__int64)&v19, v18);
    v2 = v21;
    v4 = v20;
    if ( v17 == v16 )
      break;
    while ( *v17 == -16 || *v17 == -8 )
    {
      v17 += 2;
      if ( v16 == v17 )
        goto LABEL_36;
    }
  }
  while ( v16 != v17 );
LABEL_36:
  if ( v21 == v20 )
  {
LABEL_3:
    v5 = &v2[8 * HIDWORD(v22)];
    if ( v2 == v5 )
      goto LABEL_9;
    goto LABEL_4;
  }
  v5 = &v21[8 * (unsigned int)v22];
  if ( v21 == v5 )
  {
LABEL_8:
    _libc_free((unsigned __int64)v2);
    goto LABEL_9;
  }
LABEL_4:
  v6 = v2;
  while ( 1 )
  {
    v7 = *v6;
    v8 = v6;
    if ( *v6 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v5 == (_BYTE *)++v6 )
      goto LABEL_7;
  }
  if ( v5 != (_BYTE *)v6 )
  {
    do
    {
      if ( v7 )
      {
        j___libc_free_0(*(_QWORD *)(v7 + 24));
        j_j___libc_free_0(v7, 64);
      }
      v14 = v8 + 1;
      if ( v8 + 1 == (_QWORD *)v5 )
        break;
      while ( 1 )
      {
        v7 = *v14;
        v8 = v14;
        if ( *v14 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v5 == (_BYTE *)++v14 )
          goto LABEL_23;
      }
    }
    while ( v5 != (_BYTE *)v14 );
LABEL_23:
    v2 = v21;
    v4 = v20;
  }
LABEL_7:
  if ( v4 != v2 )
    goto LABEL_8;
LABEL_9:
  v9 = *(unsigned int *)(a1 + 104);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD **)(a1 + 88);
    v11 = &v10[8 * v9];
    do
    {
      if ( *v10 != -16 && *v10 != -8 )
      {
        v12 = v10[3];
        if ( v12 != v10[2] )
          _libc_free(v12);
      }
      v10 += 8;
    }
    while ( v11 != v10 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 88));
  return j___libc_free_0(*(_QWORD *)(a1 + 56));
}
