// Function: sub_CE6110
// Address: 0xce6110
//
_BYTE *__fastcall sub_CE6110(_QWORD *a1)
{
  __int64 v1; // rdx
  __int64 v2; // rsi
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned __int64 v12; // rsi
  _BYTE *v13; // rcx
  __int64 v14; // rdx
  _BYTE *v15; // r8
  _BYTE *v16; // rax
  bool v17; // zf
  __int64 v18; // rsi
  _BYTE *v19; // r12
  _BYTE *v20; // r13
  _BYTE *result; // rax
  __int64 v22; // r12
  __int64 i; // r13
  __int64 v24; // rsi
  _QWORD v25[54]; // [rsp+20h] [rbp-A50h] BYREF
  char v26[8]; // [rsp+1D0h] [rbp-8A0h] BYREF
  __int64 v27; // [rsp+1D8h] [rbp-898h]
  char v28; // [rsp+1ECh] [rbp-884h]
  char *v29; // [rsp+230h] [rbp-840h]
  char v30; // [rsp+240h] [rbp-830h] BYREF
  char v31[8]; // [rsp+380h] [rbp-6F0h] BYREF
  __int64 v32; // [rsp+388h] [rbp-6E8h]
  char v33; // [rsp+39Ch] [rbp-6D4h]
  _BYTE *v34; // [rsp+3E0h] [rbp-690h]
  unsigned int v35; // [rsp+3E8h] [rbp-688h]
  _BYTE v36[320]; // [rsp+3F0h] [rbp-680h] BYREF
  char v37[8]; // [rsp+530h] [rbp-540h] BYREF
  __int64 v38; // [rsp+538h] [rbp-538h]
  char v39; // [rsp+54Ch] [rbp-524h]
  _BYTE *v40; // [rsp+590h] [rbp-4E0h]
  unsigned int v41; // [rsp+598h] [rbp-4D8h]
  _BYTE v42[320]; // [rsp+5A0h] [rbp-4D0h] BYREF
  _BYTE *v43; // [rsp+6E0h] [rbp-390h] BYREF
  __int64 v44; // [rsp+6E8h] [rbp-388h]
  _BYTE v45[80]; // [rsp+6F0h] [rbp-380h] BYREF
  char *v46; // [rsp+740h] [rbp-330h]
  char v47; // [rsp+750h] [rbp-320h] BYREF
  char v48[8]; // [rsp+890h] [rbp-1E0h] BYREF
  __int64 v49; // [rsp+898h] [rbp-1D8h]
  char v50; // [rsp+8ACh] [rbp-1C4h]
  char *v51; // [rsp+8F0h] [rbp-180h]
  char v52; // [rsp+900h] [rbp-170h] BYREF

  v1 = *a1;
  memset(v25, 0, sizeof(v25));
  HIDWORD(v25[13]) = 8;
  v25[1] = &v25[4];
  v25[12] = &v25[14];
  v2 = *(_QWORD *)(v1 + 80);
  LODWORD(v25[2]) = 8;
  if ( v2 )
    v2 -= 24;
  BYTE4(v25[3]) = 1;
  sub_CE3280((__int64)v26, v2);
  sub_CE35F0((__int64)v37, (__int64)v25);
  sub_CE35F0((__int64)v31, (__int64)v26);
  sub_CE35F0((__int64)&v43, (__int64)v31);
  sub_CE35F0((__int64)v48, (__int64)v37);
  if ( v34 != v36 )
    _libc_free(v34, v37);
  if ( !v33 )
    _libc_free(v32, v37);
  if ( v40 != v42 )
    _libc_free(v40, v37);
  if ( !v39 )
    _libc_free(v38, v37);
  if ( v29 != &v30 )
    _libc_free(v29, v37);
  if ( !v28 )
    _libc_free(v27, v37);
  if ( (_QWORD *)v25[12] != &v25[14] )
    _libc_free(v25[12], v37);
  if ( !BYTE4(v25[3]) )
    _libc_free(v25[1], v37);
  sub_CE3710((__int64)v31, (__int64)&v43, v3, v4, v5, v6);
  sub_CE3710((__int64)v37, (__int64)v48, v7, v8, v9, v10);
LABEL_20:
  v11 = v35;
  while ( 1 )
  {
    v12 = v41;
    v13 = v34;
    v14 = 40 * v11;
    if ( v11 == v41 )
      break;
LABEL_25:
    sub_CE4F60((__int64)a1, *(_QWORD *)&v34[v14 - 8]);
    v17 = v35 == 1;
    v11 = --v35;
    if ( !v17 )
    {
      sub_CE27D0((__int64)v31);
      goto LABEL_20;
    }
  }
  v15 = v40;
  if ( v34 != &v34[v14] )
  {
    v12 = (unsigned __int64)v40;
    v16 = v34;
    while ( *((_QWORD *)v16 + 4) == *(_QWORD *)(v12 + 32)
         && *((_DWORD *)v16 + 6) == *(_DWORD *)(v12 + 24)
         && *((_DWORD *)v16 + 2) == *(_DWORD *)(v12 + 8) )
    {
      v16 += 40;
      v12 += 40LL;
      if ( &v34[v14] == v16 )
        goto LABEL_30;
    }
    goto LABEL_25;
  }
LABEL_30:
  if ( v40 != v42 )
    _libc_free(v40, v12);
  if ( !v39 )
    _libc_free(v38, v12);
  if ( v34 != v36 )
    _libc_free(v34, v12);
  if ( !v33 )
    _libc_free(v32, v12);
  if ( v51 != &v52 )
    _libc_free(v51, v12);
  if ( !v50 )
    _libc_free(v49, v12);
  if ( v46 != &v47 )
    _libc_free(v46, v12);
  if ( !v45[12] )
    _libc_free(v44, v12);
  v18 = a1[1];
  sub_D47CF0(&v43, v18, v14, v13, v15);
  v19 = v43;
  v20 = &v43[8 * (unsigned int)v44];
  if ( v20 != v43 )
  {
    do
    {
      v18 = *(_QWORD *)v19;
      v19 += 8;
      sub_CE5730((__int64)a1, v18);
    }
    while ( v20 != v19 );
    v19 = v43;
  }
  result = v45;
  if ( v19 != v45 )
    result = (_BYTE *)_libc_free(v19, v18);
  v22 = *(_QWORD *)(*a1 + 80LL);
  for ( i = *a1 + 72LL; i != v22; v22 = *(_QWORD *)(v22 + 8) )
  {
    v24 = v22 - 24;
    if ( !v22 )
      v24 = 0;
    result = (_BYTE *)sub_CE4CE0((__int64)a1, v24);
  }
  return result;
}
