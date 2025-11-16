// Function: sub_CE3B30
// Address: 0xce3b30
//
_BYTE *__fastcall sub_CE3B30(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _BYTE *result; // rax
  _BYTE v14[8]; // [rsp+10h] [rbp-A50h] BYREF
  __int64 v15; // [rsp+18h] [rbp-A48h]
  char v16; // [rsp+2Ch] [rbp-A34h]
  _BYTE *v17; // [rsp+70h] [rbp-9F0h]
  _BYTE v18[320]; // [rsp+80h] [rbp-9E0h] BYREF
  _QWORD v19[54]; // [rsp+1C0h] [rbp-8A0h] BYREF
  _BYTE v20[8]; // [rsp+370h] [rbp-6F0h] BYREF
  __int64 v21; // [rsp+378h] [rbp-6E8h]
  char v22; // [rsp+38Ch] [rbp-6D4h]
  char *v23; // [rsp+3D0h] [rbp-690h]
  char v24; // [rsp+3E0h] [rbp-680h] BYREF
  _BYTE v25[8]; // [rsp+520h] [rbp-540h] BYREF
  __int64 v26; // [rsp+528h] [rbp-538h]
  char v27; // [rsp+53Ch] [rbp-524h]
  char *v28; // [rsp+580h] [rbp-4E0h]
  char v29; // [rsp+590h] [rbp-4D0h] BYREF
  _BYTE v30[8]; // [rsp+6D0h] [rbp-390h] BYREF
  __int64 v31; // [rsp+6D8h] [rbp-388h]
  char v32; // [rsp+6ECh] [rbp-374h]
  char *v33; // [rsp+730h] [rbp-330h]
  char v34; // [rsp+740h] [rbp-320h] BYREF
  _BYTE v35[8]; // [rsp+880h] [rbp-1E0h] BYREF
  __int64 v36; // [rsp+888h] [rbp-1D8h]
  char v37; // [rsp+89Ch] [rbp-1C4h]
  char *v38; // [rsp+8E0h] [rbp-180h]
  char v39; // [rsp+8F0h] [rbp-170h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  memset(v19, 0, sizeof(v19));
  LODWORD(v19[2]) = 8;
  v19[1] = &v19[4];
  if ( v2 )
    v2 -= 24;
  BYTE4(v19[3]) = 1;
  v19[12] = &v19[14];
  HIDWORD(v19[13]) = 8;
  sub_CE3280((__int64)v14, v2);
  sub_CE3710((__int64)v30, (__int64)v19, v3, v4, v5, v6);
  sub_CE35F0((__int64)v35, (__int64)v30);
  sub_CE3710((__int64)v20, (__int64)v14, v7, v8, (__int64)v14, v9);
  sub_CE35F0((__int64)v25, (__int64)v20);
  sub_CE37E0((__int64)v25, (__int64)v35, a1, v10, v11, v12);
  if ( v28 != &v29 )
    _libc_free(v28, v35);
  if ( !v27 )
    _libc_free(v26, v35);
  if ( v23 != &v24 )
    _libc_free(v23, v35);
  if ( !v22 )
    _libc_free(v21, v35);
  if ( v38 != &v39 )
    _libc_free(v38, v35);
  if ( !v37 )
    _libc_free(v36, v35);
  if ( v33 != &v34 )
    _libc_free(v33, v35);
  if ( !v32 )
    _libc_free(v31, v35);
  result = v18;
  if ( v17 != v18 )
    result = (_BYTE *)_libc_free(v17, v35);
  if ( !v16 )
    result = (_BYTE *)_libc_free(v15, v35);
  if ( (_QWORD *)v19[12] != &v19[14] )
    result = (_BYTE *)_libc_free(v19[12], v35);
  if ( !BYTE4(v19[3]) )
    return (_BYTE *)_libc_free(v19[1], v35);
  return result;
}
