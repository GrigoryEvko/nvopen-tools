// Function: sub_22ED620
// Address: 0x22ed620
//
void __fastcall sub_22ED620(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  int v5; // eax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned __int64 v16; // [rsp+8h] [rbp-A58h]
  __int64 v17; // [rsp+10h] [rbp-A50h] BYREF
  __int64 *v18; // [rsp+18h] [rbp-A48h]
  int v19; // [rsp+20h] [rbp-A40h]
  int v20; // [rsp+24h] [rbp-A3Ch]
  int v21; // [rsp+28h] [rbp-A38h]
  char v22; // [rsp+2Ch] [rbp-A34h]
  __int64 v23; // [rsp+30h] [rbp-A30h] BYREF
  unsigned __int64 *v24; // [rsp+70h] [rbp-9F0h]
  __int64 v25; // [rsp+78h] [rbp-9E8h]
  unsigned __int64 v26; // [rsp+80h] [rbp-9E0h] BYREF
  int v27; // [rsp+88h] [rbp-9D8h]
  unsigned __int64 v28; // [rsp+90h] [rbp-9D0h]
  int v29; // [rsp+98h] [rbp-9C8h]
  __int64 v30; // [rsp+A0h] [rbp-9C0h]
  unsigned __int64 v31[54]; // [rsp+1C0h] [rbp-8A0h] BYREF
  char v32[8]; // [rsp+370h] [rbp-6F0h] BYREF
  unsigned __int64 v33; // [rsp+378h] [rbp-6E8h]
  char v34; // [rsp+38Ch] [rbp-6D4h]
  char *v35; // [rsp+3D0h] [rbp-690h]
  char v36; // [rsp+3E0h] [rbp-680h] BYREF
  char v37[8]; // [rsp+520h] [rbp-540h] BYREF
  unsigned __int64 v38; // [rsp+528h] [rbp-538h]
  char v39; // [rsp+53Ch] [rbp-524h]
  char *v40; // [rsp+580h] [rbp-4E0h]
  char v41; // [rsp+590h] [rbp-4D0h] BYREF
  char v42[8]; // [rsp+6D0h] [rbp-390h] BYREF
  unsigned __int64 v43; // [rsp+6D8h] [rbp-388h]
  char v44; // [rsp+6ECh] [rbp-374h]
  char *v45; // [rsp+730h] [rbp-330h]
  char v46; // [rsp+740h] [rbp-320h] BYREF
  char v47[8]; // [rsp+880h] [rbp-1E0h] BYREF
  unsigned __int64 v48; // [rsp+888h] [rbp-1D8h]
  char v49; // [rsp+89Ch] [rbp-1C4h]
  char *v50; // [rsp+8E0h] [rbp-180h]
  char v51; // [rsp+8F0h] [rbp-170h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  v24 = &v26;
  memset(v31, 0, sizeof(v31));
  v31[12] = (unsigned __int64)&v31[14];
  v31[1] = (unsigned __int64)&v31[4];
  if ( v2 )
    v2 -= 24;
  HIDWORD(v31[13]) = 8;
  v18 = &v23;
  v25 = 0x800000000LL;
  v3 = *(_QWORD *)(v2 + 48);
  LODWORD(v31[2]) = 8;
  v4 = v3 & 0xFFFFFFFFFFFFFFF8LL;
  BYTE4(v31[3]) = 1;
  v19 = 8;
  v21 = 0;
  v22 = 1;
  v20 = 1;
  v23 = v2;
  v17 = 1;
  if ( v4 == v2 + 48 )
    goto LABEL_31;
  if ( !v4 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
  {
LABEL_31:
    v5 = 0;
    v7 = 0;
    v6 = 0;
  }
  else
  {
    v16 = v4 - 24;
    v5 = sub_B46E30(v4 - 24);
    v6 = v16;
    v7 = v16;
  }
  v26 = v7;
  v27 = v5;
  v28 = v6;
  v30 = v2;
  v29 = 0;
  LODWORD(v25) = 1;
  sub_E36520((__int64)&v17);
  sub_E36940((__int64)v42, (__int64)v31, v8, v9, (__int64)v42, v10);
  sub_E36820((__int64)v47, (__int64)v42);
  sub_E36940((__int64)v32, (__int64)&v17, v11, v12, (__int64)v32, (__int64)&v17);
  sub_E36820((__int64)v37, (__int64)v32);
  sub_E36A10((__int64)v37, (__int64)v47, a1, v13, v14, v15);
  if ( v40 != &v41 )
    _libc_free((unsigned __int64)v40);
  if ( !v39 )
    _libc_free(v38);
  if ( v35 != &v36 )
    _libc_free((unsigned __int64)v35);
  if ( !v34 )
    _libc_free(v33);
  if ( v50 != &v51 )
    _libc_free((unsigned __int64)v50);
  if ( !v49 )
    _libc_free(v48);
  if ( v45 != &v46 )
    _libc_free((unsigned __int64)v45);
  if ( !v44 )
    _libc_free(v43);
  if ( v24 != &v26 )
    _libc_free((unsigned __int64)v24);
  if ( !v22 )
    _libc_free((unsigned __int64)v18);
  if ( (unsigned __int64 *)v31[12] != &v31[14] )
    _libc_free(v31[12]);
  if ( !BYTE4(v31[3]) )
    _libc_free(v31[1]);
}
