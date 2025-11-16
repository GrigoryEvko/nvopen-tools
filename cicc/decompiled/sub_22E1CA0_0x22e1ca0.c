// Function: sub_22E1CA0
// Address: 0x22e1ca0
//
void __fastcall sub_22E1CA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rsi
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rdx
  _QWORD *v27; // rcx
  _QWORD *v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  bool v34; // zf
  unsigned __int64 v35[38]; // [rsp+30h] [rbp-750h] BYREF
  __int64 v36; // [rsp+160h] [rbp-620h] BYREF
  __int64 *v37; // [rsp+168h] [rbp-618h]
  int v38; // [rsp+170h] [rbp-610h]
  int v39; // [rsp+174h] [rbp-60Ch]
  int v40; // [rsp+178h] [rbp-608h]
  char v41; // [rsp+17Ch] [rbp-604h]
  __int64 v42; // [rsp+180h] [rbp-600h] BYREF
  __int64 *v43; // [rsp+1C0h] [rbp-5C0h]
  __int64 v44; // [rsp+1C8h] [rbp-5B8h]
  __int64 v45[24]; // [rsp+1D0h] [rbp-5B0h] BYREF
  char v46[8]; // [rsp+290h] [rbp-4F0h] BYREF
  unsigned __int64 v47; // [rsp+298h] [rbp-4E8h]
  char v48; // [rsp+2ACh] [rbp-4D4h]
  char v49[64]; // [rsp+2B0h] [rbp-4D0h] BYREF
  _BYTE *v50; // [rsp+2F0h] [rbp-490h] BYREF
  __int64 v51; // [rsp+2F8h] [rbp-488h]
  _BYTE v52[192]; // [rsp+300h] [rbp-480h] BYREF
  char v53[8]; // [rsp+3C0h] [rbp-3C0h] BYREF
  unsigned __int64 v54; // [rsp+3C8h] [rbp-3B8h]
  char v55; // [rsp+3DCh] [rbp-3A4h]
  char v56[64]; // [rsp+3E0h] [rbp-3A0h] BYREF
  _BYTE *v57; // [rsp+420h] [rbp-360h] BYREF
  __int64 v58; // [rsp+428h] [rbp-358h]
  _BYTE v59[192]; // [rsp+430h] [rbp-350h] BYREF
  char v60[8]; // [rsp+4F0h] [rbp-290h] BYREF
  unsigned __int64 v61; // [rsp+4F8h] [rbp-288h]
  char v62; // [rsp+50Ch] [rbp-274h]
  char *v63; // [rsp+550h] [rbp-230h] BYREF
  int v64; // [rsp+558h] [rbp-228h]
  char v65; // [rsp+560h] [rbp-220h] BYREF
  char v66[8]; // [rsp+620h] [rbp-160h] BYREF
  unsigned __int64 v67; // [rsp+628h] [rbp-158h]
  char v68; // [rsp+63Ch] [rbp-144h]
  char *v69; // [rsp+680h] [rbp-100h] BYREF
  unsigned int v70; // [rsp+688h] [rbp-F8h]
  char v71; // [rsp+690h] [rbp-F0h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  v8 = *(_QWORD *)(a1 + 8);
  if ( v7 )
  {
    v9 = (unsigned int)(*(_DWORD *)(v7 + 20) + 1);
    v10 = *(_DWORD *)(v7 + 20) + 1;
  }
  else
  {
    v9 = 0;
    v10 = 0;
  }
  v11 = 0;
  if ( v10 < *(_DWORD *)(v8 + 32) )
    v11 = *(_QWORD *)(*(_QWORD *)(v8 + 24) + 8 * v9);
  v42 = v11;
  memset(v35, 0, sizeof(v35));
  HIDWORD(v35[13]) = 8;
  v35[1] = (unsigned __int64)&v35[4];
  v35[12] = (unsigned __int64)&v35[14];
  v37 = &v42;
  v43 = v45;
  v44 = 0x800000000LL;
  v12 = *(unsigned int *)(v11 + 32);
  BYTE4(v35[3]) = 1;
  v40 = 0;
  v41 = 1;
  v13 = *(_QWORD *)(v11 + 24);
  LODWORD(v35[2]) = 8;
  v45[1] = v13;
  v45[0] = v13 + 8 * v12;
  v45[2] = v11;
  v38 = 8;
  v39 = 1;
  v36 = 1;
  LODWORD(v44) = 1;
  sub_22DCB80((__int64)&v36, v9, v11, v45[0], a5, a6);
  sub_22DD690((__int64)v53, (__int64)v35);
  sub_22DD690((__int64)v46, (__int64)&v36);
  sub_22DD690((__int64)v60, (__int64)v46);
  sub_22DD690((__int64)v66, (__int64)v53);
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
  if ( !v48 )
    _libc_free(v47);
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
  if ( !v55 )
    _libc_free(v54);
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
  if ( !v41 )
    _libc_free((unsigned __int64)v37);
  if ( (unsigned __int64 *)v35[12] != &v35[14] )
    _libc_free(v35[12]);
  if ( !BYTE4(v35[3]) )
    _libc_free(v35[1]);
  sub_C8CD80((__int64)v46, (__int64)v49, (__int64)v60, v14, v15, v16);
  v50 = v52;
  v51 = 0x800000000LL;
  if ( v64 )
    sub_22DD7B0((__int64)&v50, (__int64 *)&v63, v17, v18, v19, v20);
  sub_C8CD80((__int64)v53, (__int64)v56, (__int64)v66, v18, v19, v20);
  v24 = v70;
  v57 = v59;
  v58 = 0x800000000LL;
  if ( v70 )
  {
    sub_22DD7B0((__int64)&v57, (__int64 *)&v69, v21, v70, v22, v23);
    v24 = (unsigned int)v58;
  }
LABEL_25:
  v25 = (unsigned int)v51;
  while ( 1 )
  {
    v26 = 24 * v25;
    if ( v25 != v24 )
      goto LABEL_30;
    if ( &v50[v26] == v50 )
      break;
    v27 = v57;
    v28 = v50;
    while ( v28[2] == v27[2] && v28[1] == v27[1] && *v28 == *v27 )
    {
      v28 += 3;
      v27 += 3;
      if ( &v50[v26] == (_BYTE *)v28 )
        goto LABEL_36;
    }
LABEL_30:
    v29 = **(_QWORD **)&v50[v26 - 8];
    sub_22E1B70(a1, v29, a3);
    v34 = (_DWORD)v51 == 1;
    v25 = (unsigned int)(v51 - 1);
    LODWORD(v51) = v51 - 1;
    if ( !v34 )
    {
      sub_22DCB80((__int64)v46, v29, v30, v31, v32, v33);
      v24 = (unsigned int)v58;
      goto LABEL_25;
    }
    v24 = (unsigned int)v58;
  }
LABEL_36:
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
  if ( !v55 )
    _libc_free(v54);
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
  if ( !v48 )
    _libc_free(v47);
  if ( v69 != &v71 )
    _libc_free((unsigned __int64)v69);
  if ( !v68 )
    _libc_free(v67);
  if ( v63 != &v65 )
    _libc_free((unsigned __int64)v63);
  if ( !v62 )
    _libc_free(v61);
}
