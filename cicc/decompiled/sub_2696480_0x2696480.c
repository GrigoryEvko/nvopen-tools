// Function: sub_2696480
// Address: 0x2696480
//
void __fastcall sub_2696480(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  int v6; // eax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int64 v17; // [rsp+8h] [rbp-A58h]
  __int64 v18; // [rsp+10h] [rbp-A50h] BYREF
  __int64 *v19; // [rsp+18h] [rbp-A48h]
  int v20; // [rsp+20h] [rbp-A40h]
  int v21; // [rsp+24h] [rbp-A3Ch]
  int v22; // [rsp+28h] [rbp-A38h]
  char v23; // [rsp+2Ch] [rbp-A34h]
  __int64 v24; // [rsp+30h] [rbp-A30h] BYREF
  unsigned __int64 *v25; // [rsp+70h] [rbp-9F0h]
  __int64 v26; // [rsp+78h] [rbp-9E8h]
  unsigned __int64 v27; // [rsp+80h] [rbp-9E0h] BYREF
  int v28; // [rsp+88h] [rbp-9D8h]
  unsigned __int64 v29; // [rsp+90h] [rbp-9D0h]
  int v30; // [rsp+98h] [rbp-9C8h]
  __int64 v31; // [rsp+A0h] [rbp-9C0h]
  unsigned __int64 v32[54]; // [rsp+1C0h] [rbp-8A0h] BYREF
  char v33[8]; // [rsp+370h] [rbp-6F0h] BYREF
  unsigned __int64 v34; // [rsp+378h] [rbp-6E8h]
  char v35; // [rsp+38Ch] [rbp-6D4h]
  char *v36; // [rsp+3D0h] [rbp-690h]
  char v37; // [rsp+3E0h] [rbp-680h] BYREF
  char v38[8]; // [rsp+520h] [rbp-540h] BYREF
  unsigned __int64 v39; // [rsp+528h] [rbp-538h]
  char v40; // [rsp+53Ch] [rbp-524h]
  char *v41; // [rsp+580h] [rbp-4E0h]
  char v42; // [rsp+590h] [rbp-4D0h] BYREF
  char v43[8]; // [rsp+6D0h] [rbp-390h] BYREF
  unsigned __int64 v44; // [rsp+6D8h] [rbp-388h]
  char v45; // [rsp+6ECh] [rbp-374h]
  char *v46; // [rsp+730h] [rbp-330h]
  char v47; // [rsp+740h] [rbp-320h] BYREF
  char v48[8]; // [rsp+880h] [rbp-1E0h] BYREF
  unsigned __int64 v49; // [rsp+888h] [rbp-1D8h]
  char v50; // [rsp+89Ch] [rbp-1C4h]
  char *v51; // [rsp+8E0h] [rbp-180h]
  char v52; // [rsp+8F0h] [rbp-170h] BYREF

  v23 = 1;
  memset(v32, 0, sizeof(v32));
  v32[12] = (unsigned __int64)&v32[14];
  v32[1] = (unsigned __int64)&v32[4];
  v2 = *a2;
  HIDWORD(v32[13]) = 8;
  v3 = *(_QWORD *)(v2 + 80);
  v25 = &v27;
  LODWORD(v32[2]) = 8;
  BYTE4(v32[3]) = 1;
  v20 = 8;
  if ( v3 )
    v3 -= 24;
  v19 = &v24;
  v26 = 0x800000000LL;
  v4 = *(_QWORD *)(v3 + 48);
  v22 = 0;
  v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v21 = 1;
  v24 = v3;
  v18 = 1;
  if ( v5 == v3 + 48 )
    goto LABEL_31;
  if ( !v5 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
  {
LABEL_31:
    v6 = 0;
    v8 = 0;
    v7 = 0;
  }
  else
  {
    v17 = v5 - 24;
    v6 = sub_B46E30(v5 - 24);
    v7 = v17;
    v8 = v17;
  }
  v27 = v8;
  v28 = v6;
  v29 = v7;
  v31 = v3;
  v30 = 0;
  LODWORD(v26) = 1;
  sub_CE27D0((__int64)&v18);
  sub_CE3710((__int64)v43, (__int64)v32, v9, v10, (__int64)v43, v11);
  sub_CE35F0((__int64)v48, (__int64)v43);
  sub_CE3710((__int64)v33, (__int64)&v18, v12, v13, (__int64)v33, (__int64)&v18);
  sub_CE35F0((__int64)v38, (__int64)v33);
  sub_CE37E0((__int64)v38, (__int64)v48, a1, v14, v15, v16);
  if ( v41 != &v42 )
    _libc_free((unsigned __int64)v41);
  if ( !v40 )
    _libc_free(v39);
  if ( v36 != &v37 )
    _libc_free((unsigned __int64)v36);
  if ( !v35 )
    _libc_free(v34);
  if ( v51 != &v52 )
    _libc_free((unsigned __int64)v51);
  if ( !v50 )
    _libc_free(v49);
  if ( v46 != &v47 )
    _libc_free((unsigned __int64)v46);
  if ( !v45 )
    _libc_free(v44);
  if ( v25 != &v27 )
    _libc_free((unsigned __int64)v25);
  if ( !v23 )
    _libc_free((unsigned __int64)v19);
  if ( (unsigned __int64 *)v32[12] != &v32[14] )
    _libc_free(v32[12]);
  if ( !BYTE4(v32[3]) )
    _libc_free(v32[1]);
}
