// Function: sub_13BB090
// Address: 0x13bb090
//
void __fastcall sub_13BB090(__int64 a1)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // rsi
  _QWORD *v4; // rdi
  __int64 v5; // rdx
  _QWORD *v6; // rsi
  _QWORD *v7; // r8
  unsigned __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  _QWORD *v12; // rax
  char v13; // cl
  __int64 v14; // rcx
  __int64 v15; // r8
  unsigned __int64 v16; // r13
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  char v21; // si
  unsigned __int64 v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v24; // rax
  char v25; // si
  char v26; // r8
  _QWORD v27[16]; // [rsp+0h] [rbp-230h] BYREF
  _QWORD v28[16]; // [rsp+80h] [rbp-1B0h] BYREF
  _QWORD v29[2]; // [rsp+100h] [rbp-130h] BYREF
  unsigned __int64 v30; // [rsp+110h] [rbp-120h]
  _QWORD *v31; // [rsp+168h] [rbp-C8h]
  _QWORD *v32; // [rsp+170h] [rbp-C0h]
  __int64 v33; // [rsp+178h] [rbp-B8h]
  char v34[8]; // [rsp+180h] [rbp-B0h] BYREF
  __int64 v35; // [rsp+188h] [rbp-A8h]
  unsigned __int64 v36; // [rsp+190h] [rbp-A0h]
  __int64 v37; // [rsp+1E8h] [rbp-48h]
  __int64 v38; // [rsp+1F0h] [rbp-40h]
  __int64 v39; // [rsp+1F8h] [rbp-38h]

  v2 = **(_QWORD **)(a1 + 8);
  memset(v27, 0, sizeof(v27));
  v27[1] = &v27[5];
  v27[2] = &v27[5];
  LODWORD(v27[3]) = 8;
  v3 = *(_QWORD *)(v2 + 80);
  if ( v3 )
  {
    sub_13B83E0((__int64)v28, v3);
  }
  else
  {
    memset(v28, 0, sizeof(v28));
    LODWORD(v28[3]) = 8;
    v28[1] = &v28[5];
    v28[2] = &v28[5];
  }
  sub_13BA6D0(v29, v28, v27);
  if ( v28[13] )
    j_j___libc_free_0(v28[13], v28[15] - v28[13]);
  if ( v28[2] != v28[1] )
    _libc_free(v28[2]);
  if ( v27[13] )
    j_j___libc_free_0(v27[13], v27[15] - v27[13]);
  if ( v27[2] != v27[1] )
    _libc_free(v27[2]);
  v4 = v27;
  sub_16CCCB0(v27, &v27[5], v29);
  v6 = v32;
  v7 = v31;
  memset(&v27[13], 0, 24);
  v8 = (char *)v32 - (char *)v31;
  if ( v32 == v31 )
  {
    v10 = 0;
  }
  else
  {
    if ( v8 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_65;
    v9 = sub_22077B0((char *)v32 - (char *)v31);
    v6 = v32;
    v7 = v31;
    v10 = v9;
  }
  v27[13] = v10;
  v27[14] = v10;
  v27[15] = v10 + v8;
  if ( v7 != v6 )
  {
    v11 = v10;
    v12 = v7;
    do
    {
      if ( v11 )
      {
        *(_QWORD *)v11 = *v12;
        v13 = *((_BYTE *)v12 + 16);
        *(_BYTE *)(v11 + 16) = v13;
        if ( v13 )
          *(_QWORD *)(v11 + 8) = v12[1];
      }
      v12 += 3;
      v11 += 24;
    }
    while ( v12 != v6 );
    v10 += 8 * ((unsigned __int64)((char *)(v12 - 3) - (char *)v7) >> 3) + 24;
  }
  v27[14] = v10;
  v4 = v28;
  v6 = &v28[5];
  sub_16CCCB0(v28, &v28[5], v34);
  v14 = v38;
  v15 = v37;
  memset(&v28[13], 0, 24);
  v16 = v38 - v37;
  if ( v38 == v37 )
  {
    v18 = 0;
    goto LABEL_24;
  }
  if ( v16 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_65:
    sub_4261EA(v4, v6, v5);
  v17 = sub_22077B0(v38 - v37);
  v14 = v38;
  v15 = v37;
  v18 = v17;
LABEL_24:
  v28[13] = v18;
  v28[14] = v18;
  v28[15] = v18 + v16;
  if ( v15 == v14 )
  {
    v22 = v18;
  }
  else
  {
    v19 = v18;
    v20 = v15;
    do
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = *(_QWORD *)v20;
        v21 = *(_BYTE *)(v20 + 16);
        *(_BYTE *)(v19 + 16) = v21;
        if ( v21 )
          *(_QWORD *)(v19 + 8) = *(_QWORD *)(v20 + 8);
      }
      v20 += 24;
      v19 += 24LL;
    }
    while ( v14 != v20 );
    v22 = v18 + 8 * ((unsigned __int64)(v14 - 24 - v15) >> 3) + 24;
  }
  for ( v28[14] = v22; ; v22 = v28[14] )
  {
    v23 = v27[13];
    if ( v27[14] - v27[13] != v22 - v18 )
      goto LABEL_32;
    if ( v27[13] == v27[14] )
      break;
    v24 = v18;
    while ( *(_QWORD *)v23 == *(_QWORD *)v24 )
    {
      v25 = *(_BYTE *)(v23 + 16);
      v26 = *(_BYTE *)(v24 + 16);
      if ( v25 && v26 )
      {
        if ( *(_QWORD *)(v23 + 8) != *(_QWORD *)(v24 + 8) )
          break;
        v23 += 24;
        v24 += 24LL;
        if ( v27[14] == v23 )
          goto LABEL_41;
      }
      else
      {
        if ( v26 != v25 )
          break;
        v23 += 24;
        v24 += 24LL;
        if ( v27[14] == v23 )
          goto LABEL_41;
      }
    }
LABEL_32:
    sub_13B8B30((_BYTE *)a1, *(__int64 **)(v27[14] - 24LL));
    sub_13BA8B0((__int64)v27);
    v18 = v28[13];
  }
LABEL_41:
  if ( v18 )
    j_j___libc_free_0(v18, v28[15] - v18);
  if ( v28[2] != v28[1] )
    _libc_free(v28[2]);
  if ( v27[13] )
    j_j___libc_free_0(v27[13], v27[15] - v27[13]);
  if ( v27[2] != v27[1] )
    _libc_free(v27[2]);
  if ( v37 )
    j_j___libc_free_0(v37, v39 - v37);
  if ( v36 != v35 )
    _libc_free(v36);
  if ( v31 )
    j_j___libc_free_0(v31, v33 - (_QWORD)v31);
  if ( v30 != v29[1] )
    _libc_free(v30);
}
