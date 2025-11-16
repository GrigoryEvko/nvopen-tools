// Function: sub_1EDBA90
// Address: 0x1edba90
//
void __fastcall sub_1EDBA90(__int64 a1, __int64 a2, __int64 a3, int a4, _DWORD *a5, int a6)
{
  unsigned __int64 v10; // r14
  int v11; // ecx
  int v12; // esi
  __int64 v13; // rax
  __int64 v14; // rdx
  _BYTE *v15; // rdi
  unsigned __int64 v16; // r14
  _BYTE *v17; // rax
  _BYTE *i; // rdx
  int v19; // ecx
  int v20; // esi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rbx
  _BYTE *v25; // rdi
  __int64 v26; // rcx
  unsigned __int64 v27; // r13
  _BYTE *v28; // rax
  _BYTE *j; // rdx
  __int64 v30; // rbx
  __int64 k; // r13
  unsigned int v32; // esi
  __int64 v33; // r13
  __int64 v34; // rbx
  unsigned int v35; // esi
  __int64 v36; // r9
  __int64 *v39; // [rsp+50h] [rbp-490h] BYREF
  __int64 v40; // [rsp+58h] [rbp-488h]
  _BYTE v41[64]; // [rsp+60h] [rbp-480h] BYREF
  unsigned __int64 v42[2]; // [rsp+A0h] [rbp-440h] BYREF
  _BYTE v43[128]; // [rsp+B0h] [rbp-430h] BYREF
  __int64 v44; // [rsp+130h] [rbp-3B0h] BYREF
  int v45; // [rsp+138h] [rbp-3A8h]
  int v46; // [rsp+13Ch] [rbp-3A4h]
  int v47; // [rsp+140h] [rbp-3A0h]
  __int16 v48; // [rsp+144h] [rbp-39Ch]
  unsigned __int64 *v49; // [rsp+148h] [rbp-398h]
  _DWORD *v50; // [rsp+150h] [rbp-390h]
  __int64 v51; // [rsp+158h] [rbp-388h]
  __int64 v52; // [rsp+160h] [rbp-380h]
  __int64 v53; // [rsp+168h] [rbp-378h]
  _BYTE *v54; // [rsp+170h] [rbp-370h] BYREF
  __int64 v55; // [rsp+178h] [rbp-368h]
  _BYTE s[32]; // [rsp+180h] [rbp-360h] BYREF
  _BYTE *v57; // [rsp+1A0h] [rbp-340h] BYREF
  __int64 v58; // [rsp+1A8h] [rbp-338h]
  _BYTE v59[320]; // [rsp+1B0h] [rbp-330h] BYREF
  __int64 v60; // [rsp+2F0h] [rbp-1F0h] BYREF
  int v61; // [rsp+2F8h] [rbp-1E8h]
  int v62; // [rsp+2FCh] [rbp-1E4h]
  int v63; // [rsp+300h] [rbp-1E0h]
  __int16 v64; // [rsp+304h] [rbp-1DCh]
  unsigned __int64 *v65; // [rsp+308h] [rbp-1D8h]
  _DWORD *v66; // [rsp+310h] [rbp-1D0h]
  __int64 v67; // [rsp+318h] [rbp-1C8h]
  __int64 v68; // [rsp+320h] [rbp-1C0h]
  __int64 v69; // [rsp+328h] [rbp-1B8h]
  _BYTE *v70; // [rsp+330h] [rbp-1B0h] BYREF
  __int64 v71; // [rsp+338h] [rbp-1A8h]
  _BYTE v72[32]; // [rsp+340h] [rbp-1A0h] BYREF
  _BYTE *v73; // [rsp+360h] [rbp-180h] BYREF
  __int64 v74; // [rsp+368h] [rbp-178h]
  _BYTE v75[368]; // [rsp+370h] [rbp-170h] BYREF

  v10 = *(unsigned int *)(a3 + 72);
  v42[0] = (unsigned __int64)v43;
  v11 = a5[5];
  v42[1] = 0x1000000000LL;
  v12 = a5[3];
  v51 = *(_QWORD *)(a1 + 272);
  v13 = *(_QWORD *)(v51 + 272);
  v14 = *(_QWORD *)(a1 + 256);
  v44 = a3;
  v15 = s;
  v45 = v12;
  v52 = v13;
  v46 = v11;
  v47 = a4;
  v48 = 257;
  v49 = v42;
  v50 = a5;
  v53 = v14;
  v54 = s;
  v55 = 0x800000000LL;
  if ( (unsigned int)v10 > 8 )
  {
    sub_16CD150((__int64)&v54, s, v10, 4, (int)a5, a6);
    v15 = v54;
  }
  LODWORD(v55) = v10;
  if ( 4 * v10 )
    memset(v15, 255, 4 * v10);
  v58 = 0x800000000LL;
  v16 = *(unsigned int *)(a3 + 72);
  v17 = v59;
  v57 = v59;
  if ( (unsigned int)v16 > 8 )
  {
    sub_16CD150((__int64)&v57, v59, v16, 40, (int)a5, a6);
    v17 = v57;
  }
  LODWORD(v58) = v16;
  for ( i = &v17[40 * v16]; i != v17; v17 += 40 )
  {
    if ( v17 )
    {
      *(_DWORD *)v17 = 0;
      *((_DWORD *)v17 + 1) = 0;
      *((_DWORD *)v17 + 2) = 0;
      *((_QWORD *)v17 + 2) = 0;
      *((_QWORD *)v17 + 3) = 0;
      v17[32] = 0;
      v17[33] = 0;
      v17[34] = 0;
      v17[35] = 0;
    }
  }
  v19 = a5[4];
  v20 = a5[2];
  v63 = a4;
  v21 = *(_QWORD *)(a1 + 272);
  v66 = a5;
  v62 = v19;
  v22 = *(_QWORD *)(a1 + 256);
  v67 = v21;
  v23 = *(_QWORD *)(v21 + 272);
  v24 = *(unsigned int *)(a2 + 72);
  v64 = 257;
  v60 = a2;
  v25 = v72;
  v68 = v23;
  v61 = v20;
  v65 = v42;
  v69 = v22;
  v70 = v72;
  v71 = 0x800000000LL;
  if ( (unsigned int)v24 > 8 )
  {
    sub_16CD150((__int64)&v70, v72, v24, 4, (int)a5, a6);
    v25 = v70;
  }
  LODWORD(v71) = v24;
  if ( 4 * v24 )
    memset(v25, 255, 4 * v24);
  v26 = 0x800000000LL;
  v74 = 0x800000000LL;
  v27 = *(unsigned int *)(a2 + 72);
  v28 = v75;
  v73 = v75;
  if ( (unsigned int)v27 > 8 )
  {
    sub_16CD150((__int64)&v73, v75, v27, 40, (int)a5, a6);
    v28 = v73;
  }
  LODWORD(v74) = v27;
  for ( j = &v28[40 * v27]; j != v28; v28 += 40 )
  {
    if ( v28 )
    {
      *(_DWORD *)v28 = 0;
      *((_DWORD *)v28 + 1) = 0;
      *((_DWORD *)v28 + 2) = 0;
      *((_QWORD *)v28 + 2) = 0;
      *((_QWORD *)v28 + 3) = 0;
      v28[32] = 0;
      v28[33] = 0;
      v28[34] = 0;
      v28[35] = 0;
    }
  }
  v30 = *(unsigned int *)(v60 + 72);
  if ( (_DWORD)v30 )
  {
    for ( k = 0; k != v30; ++k )
    {
      v32 = k;
      sub_1EDB260((__int64)&v60, v32, (__int64)&v44, v26, (int)a5);
    }
  }
  v33 = 0;
  v34 = *(unsigned int *)(v44 + 72);
  if ( (_DWORD)v34 )
  {
    do
    {
      v35 = v33++;
      sub_1EDB260((__int64)&v44, v35, (__int64)&v60, v26, (int)a5);
    }
    while ( v33 != v34 );
  }
  sub_1EDA110((__int64)&v60, (__int64)&v44);
  sub_1EDA110((__int64)&v44, (__int64)&v60);
  v39 = (__int64 *)v41;
  v40 = 0x800000000LL;
  sub_1ED7EC0(&v60, &v44, (__int64)&v39, 0);
  sub_1ED7EC0(&v44, &v60, (__int64)&v39, 0);
  sub_1ED7E40(&v60);
  sub_1ED7E40(&v44);
  sub_1DB9000(a2, (__int64 *)a3, (__int64)v70, (__int64)v54, (__int64)v42, v36);
  if ( (_DWORD)v40 )
    sub_1DBC0D0(*(_QWORD **)(a1 + 272), a2, v39, (unsigned int)v40, 0, 0);
  if ( v39 != (__int64 *)v41 )
    _libc_free((unsigned __int64)v39);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
  if ( v54 != s )
    _libc_free((unsigned __int64)v54);
  if ( (_BYTE *)v42[0] != v43 )
    _libc_free(v42[0]);
}
