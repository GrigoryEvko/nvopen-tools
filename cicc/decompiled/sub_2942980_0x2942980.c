// Function: sub_2942980
// Address: 0x2942980
//
void __fastcall sub_2942980(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  int v4; // eax
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 v15; // [rsp+8h] [rbp-A58h]
  __int64 v16; // [rsp+10h] [rbp-A50h] BYREF
  __int64 *v17; // [rsp+18h] [rbp-A48h]
  int v18; // [rsp+20h] [rbp-A40h]
  int v19; // [rsp+24h] [rbp-A3Ch]
  int v20; // [rsp+28h] [rbp-A38h]
  char v21; // [rsp+2Ch] [rbp-A34h]
  __int64 v22; // [rsp+30h] [rbp-A30h] BYREF
  unsigned __int64 *v23; // [rsp+70h] [rbp-9F0h]
  __int64 v24; // [rsp+78h] [rbp-9E8h]
  unsigned __int64 v25; // [rsp+80h] [rbp-9E0h] BYREF
  int v26; // [rsp+88h] [rbp-9D8h]
  unsigned __int64 v27; // [rsp+90h] [rbp-9D0h]
  int v28; // [rsp+98h] [rbp-9C8h]
  __int64 v29; // [rsp+A0h] [rbp-9C0h]
  unsigned __int64 v30[54]; // [rsp+1C0h] [rbp-8A0h] BYREF
  char v31[8]; // [rsp+370h] [rbp-6F0h] BYREF
  unsigned __int64 v32; // [rsp+378h] [rbp-6E8h]
  char v33; // [rsp+38Ch] [rbp-6D4h]
  char *v34; // [rsp+3D0h] [rbp-690h]
  char v35; // [rsp+3E0h] [rbp-680h] BYREF
  char v36[8]; // [rsp+520h] [rbp-540h] BYREF
  unsigned __int64 v37; // [rsp+528h] [rbp-538h]
  char v38; // [rsp+53Ch] [rbp-524h]
  char *v39; // [rsp+580h] [rbp-4E0h]
  char v40; // [rsp+590h] [rbp-4D0h] BYREF
  char v41[8]; // [rsp+6D0h] [rbp-390h] BYREF
  unsigned __int64 v42; // [rsp+6D8h] [rbp-388h]
  char v43; // [rsp+6ECh] [rbp-374h]
  char *v44; // [rsp+730h] [rbp-330h]
  char v45; // [rsp+740h] [rbp-320h] BYREF
  char v46[8]; // [rsp+880h] [rbp-1E0h] BYREF
  unsigned __int64 v47; // [rsp+888h] [rbp-1D8h]
  char v48; // [rsp+89Ch] [rbp-1C4h]
  char *v49; // [rsp+8E0h] [rbp-180h]
  char v50; // [rsp+8F0h] [rbp-170h] BYREF

  v21 = 1;
  memset(v30, 0, sizeof(v30));
  LODWORD(v30[2]) = 8;
  v30[1] = (unsigned __int64)&v30[4];
  v17 = &v22;
  v24 = 0x800000000LL;
  v2 = *(_QWORD *)(a2 + 48);
  BYTE4(v30[3]) = 1;
  v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
  v30[12] = (unsigned __int64)&v30[14];
  HIDWORD(v30[13]) = 8;
  v18 = 8;
  v20 = 0;
  v23 = &v25;
  v19 = 1;
  v22 = a2;
  v16 = 1;
  if ( v3 == a2 + 48 )
    goto LABEL_29;
  if ( !v3 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA )
  {
LABEL_29:
    v4 = 0;
    v6 = 0;
    v5 = 0;
  }
  else
  {
    v15 = v3 - 24;
    v4 = sub_B46E30(v3 - 24);
    v5 = v15;
    v6 = v15;
  }
  v25 = v6;
  v26 = v4;
  v27 = v5;
  v29 = a2;
  v28 = 0;
  LODWORD(v24) = 1;
  sub_D4D230((__int64)&v16);
  sub_F1FCA0((__int64)v41, (__int64)v30, v7, v8, (__int64)v41, v9);
  sub_F1FB80((__int64)v46, (__int64)v41);
  sub_F1FCA0((__int64)v31, (__int64)&v16, v10, v11, (__int64)v31, (__int64)&v16);
  sub_F1FB80((__int64)v36, (__int64)v31);
  sub_F1FD70((__int64)v36, (__int64)v46, a1, v12, v13, v14);
  if ( v39 != &v40 )
    _libc_free((unsigned __int64)v39);
  if ( !v38 )
    _libc_free(v37);
  if ( v34 != &v35 )
    _libc_free((unsigned __int64)v34);
  if ( !v33 )
    _libc_free(v32);
  if ( v49 != &v50 )
    _libc_free((unsigned __int64)v49);
  if ( !v48 )
    _libc_free(v47);
  if ( v44 != &v45 )
    _libc_free((unsigned __int64)v44);
  if ( !v43 )
    _libc_free(v42);
  if ( v23 != &v25 )
    _libc_free((unsigned __int64)v23);
  if ( !v21 )
    _libc_free((unsigned __int64)v17);
  if ( (unsigned __int64 *)v30[12] != &v30[14] )
    _libc_free(v30[12]);
  if ( !BYTE4(v30[3]) )
    _libc_free(v30[1]);
}
