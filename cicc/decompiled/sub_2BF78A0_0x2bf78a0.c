// Function: sub_2BF78A0
// Address: 0x2bf78a0
//
void __fastcall sub_2BF78A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool v5; // zf
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // [rsp+10h] [rbp-A50h] BYREF
  __int64 *v17; // [rsp+18h] [rbp-A48h]
  int v18; // [rsp+20h] [rbp-A40h]
  int v19; // [rsp+24h] [rbp-A3Ch]
  int v20; // [rsp+28h] [rbp-A38h]
  char v21; // [rsp+2Ch] [rbp-A34h]
  __int64 v22; // [rsp+30h] [rbp-A30h] BYREF
  _QWORD *v23; // [rsp+70h] [rbp-9F0h]
  __int64 v24; // [rsp+78h] [rbp-9E8h]
  _QWORD v25[40]; // [rsp+80h] [rbp-9E0h] BYREF
  unsigned __int64 v26[54]; // [rsp+1C0h] [rbp-8A0h] BYREF
  char v27[8]; // [rsp+370h] [rbp-6F0h] BYREF
  unsigned __int64 v28; // [rsp+378h] [rbp-6E8h]
  char v29; // [rsp+38Ch] [rbp-6D4h]
  char v30[64]; // [rsp+390h] [rbp-6D0h] BYREF
  char *v31; // [rsp+3D0h] [rbp-690h] BYREF
  int v32; // [rsp+3D8h] [rbp-688h]
  char v33; // [rsp+3E0h] [rbp-680h] BYREF
  char v34[8]; // [rsp+520h] [rbp-540h] BYREF
  unsigned __int64 v35; // [rsp+528h] [rbp-538h]
  char v36; // [rsp+53Ch] [rbp-524h]
  char v37[64]; // [rsp+540h] [rbp-520h] BYREF
  unsigned __int64 v38[2]; // [rsp+580h] [rbp-4E0h] BYREF
  _BYTE v39[320]; // [rsp+590h] [rbp-4D0h] BYREF
  char v40[8]; // [rsp+6D0h] [rbp-390h] BYREF
  unsigned __int64 v41; // [rsp+6D8h] [rbp-388h]
  char v42; // [rsp+6ECh] [rbp-374h]
  char v43[64]; // [rsp+6F0h] [rbp-370h] BYREF
  char *v44; // [rsp+730h] [rbp-330h] BYREF
  unsigned int v45; // [rsp+738h] [rbp-328h]
  char v46; // [rsp+740h] [rbp-320h] BYREF
  char v47[8]; // [rsp+880h] [rbp-1E0h] BYREF
  unsigned __int64 v48; // [rsp+888h] [rbp-1D8h]
  char v49; // [rsp+89Ch] [rbp-1C4h]
  char v50[64]; // [rsp+8A0h] [rbp-1C0h] BYREF
  unsigned __int64 v51[2]; // [rsp+8E0h] [rbp-180h] BYREF
  _BYTE v52[368]; // [rsp+8F0h] [rbp-170h] BYREF

  v5 = *(_BYTE *)(a2 + 8) == 0;
  v18 = 8;
  memset(v26, 0, sizeof(v26));
  LODWORD(v26[2]) = 8;
  v6 = 1;
  v26[1] = (unsigned __int64)&v26[4];
  v17 = &v22;
  BYTE4(v26[3]) = 1;
  v26[12] = (unsigned __int64)&v26[14];
  HIDWORD(v26[13]) = 8;
  v20 = 0;
  v21 = 1;
  v23 = v25;
  v24 = 0x800000000LL;
  v19 = 1;
  v22 = a2;
  v16 = 1;
  if ( !v5 )
  {
    v15 = a2;
    while ( 1 )
    {
      v6 = *(unsigned int *)(v15 + 88);
      if ( (_DWORD)v6 )
        break;
      v15 = *(_QWORD *)(v15 + 48);
      if ( !v15 )
      {
        v6 = 0;
        break;
      }
    }
  }
  v25[0] = a2;
  v25[1] = v6;
  v25[2] = a2;
  v25[4] = a2;
  v25[3] = 0;
  LODWORD(v24) = 1;
  sub_2BF6FC0((__int64)&v16, a2, a2, v6, a5, (__int64)&v16);
  sub_2BF6E90((__int64)v40, (__int64)v26, v7, v8, (__int64)v40, v9);
  sub_C8CF70((__int64)v47, v50, 8, (__int64)v43, (__int64)v40);
  v12 = v45;
  v51[0] = (unsigned __int64)v52;
  v51[1] = 0x800000000LL;
  if ( v45 )
    sub_2BF7200((__int64)v51, (__int64)&v44, v45, v10, v11, (__int64)&v16);
  sub_2BF6E90((__int64)v27, (__int64)&v16, v12, v10, (__int64)v27, (__int64)&v16);
  sub_C8CF70((__int64)v34, v37, 8, (__int64)v30, (__int64)v27);
  v38[1] = 0x800000000LL;
  v38[0] = (unsigned __int64)v39;
  if ( v32 )
    sub_2BF7200((__int64)v38, (__int64)&v31, v13, (__int64)v39, v14, (__int64)v34);
  sub_2BF7440((__int64)v34, (__int64)v47, a1, (__int64)v39, v14, (__int64)v34);
  if ( (_BYTE *)v38[0] != v39 )
    _libc_free(v38[0]);
  if ( !v36 )
    _libc_free(v35);
  if ( v31 != &v33 )
    _libc_free((unsigned __int64)v31);
  if ( !v29 )
    _libc_free(v28);
  if ( (_BYTE *)v51[0] != v52 )
    _libc_free(v51[0]);
  if ( !v49 )
    _libc_free(v48);
  if ( v44 != &v46 )
    _libc_free((unsigned __int64)v44);
  if ( !v42 )
    _libc_free(v41);
  if ( v23 != v25 )
    _libc_free((unsigned __int64)v23);
  if ( !v21 )
    _libc_free((unsigned __int64)v17);
  if ( (unsigned __int64 *)v26[12] != &v26[14] )
    _libc_free(v26[12]);
  if ( !BYTE4(v26[3]) )
    _libc_free(v26[1]);
}
