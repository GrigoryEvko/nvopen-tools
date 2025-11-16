// Function: sub_2DAD040
// Address: 0x2dad040
//
__int64 __fastcall sub_2DAD040(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rcx
  _QWORD *v23; // rax
  __int64 v24; // r15
  __int64 v25; // rsi
  _QWORD *v26; // r12
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 *v30; // rax
  __int64 v31; // rdx
  _QWORD *v32; // r15
  __int64 v33; // rax
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rdx
  __int64 v36; // rax
  unsigned __int64 v37; // r14
  unsigned __int64 v38; // rax
  unsigned __int8 v39; // al
  bool v40; // zf
  size_t v41; // rdx
  unsigned __int8 v43; // [rsp+28h] [rbp-758h]
  unsigned __int64 v44[38]; // [rsp+30h] [rbp-750h] BYREF
  __int64 v45; // [rsp+160h] [rbp-620h] BYREF
  __int64 *v46; // [rsp+168h] [rbp-618h]
  int v47; // [rsp+170h] [rbp-610h]
  int v48; // [rsp+174h] [rbp-60Ch]
  int v49; // [rsp+178h] [rbp-608h]
  char v50; // [rsp+17Ch] [rbp-604h]
  __int64 v51; // [rsp+180h] [rbp-600h] BYREF
  __int64 *v52; // [rsp+1C0h] [rbp-5C0h]
  int v53; // [rsp+1C8h] [rbp-5B8h]
  int v54; // [rsp+1CCh] [rbp-5B4h]
  __int64 v55[24]; // [rsp+1D0h] [rbp-5B0h] BYREF
  char v56[8]; // [rsp+290h] [rbp-4F0h] BYREF
  unsigned __int64 v57; // [rsp+298h] [rbp-4E8h]
  char v58; // [rsp+2ACh] [rbp-4D4h]
  char v59[64]; // [rsp+2B0h] [rbp-4D0h] BYREF
  _BYTE *v60; // [rsp+2F0h] [rbp-490h] BYREF
  __int64 v61; // [rsp+2F8h] [rbp-488h]
  _BYTE v62[192]; // [rsp+300h] [rbp-480h] BYREF
  char v63[8]; // [rsp+3C0h] [rbp-3C0h] BYREF
  unsigned __int64 v64; // [rsp+3C8h] [rbp-3B8h]
  char v65; // [rsp+3DCh] [rbp-3A4h]
  char v66[64]; // [rsp+3E0h] [rbp-3A0h] BYREF
  _BYTE *v67; // [rsp+420h] [rbp-360h] BYREF
  __int64 v68; // [rsp+428h] [rbp-358h]
  _BYTE v69[192]; // [rsp+430h] [rbp-350h] BYREF
  char v70[8]; // [rsp+4F0h] [rbp-290h] BYREF
  unsigned __int64 v71; // [rsp+4F8h] [rbp-288h]
  char v72; // [rsp+50Ch] [rbp-274h]
  char *v73; // [rsp+550h] [rbp-230h] BYREF
  int v74; // [rsp+558h] [rbp-228h]
  char v75; // [rsp+560h] [rbp-220h] BYREF
  char v76[8]; // [rsp+620h] [rbp-160h] BYREF
  unsigned __int64 v77; // [rsp+628h] [rbp-158h]
  char v78; // [rsp+63Ch] [rbp-144h]
  char *v79; // [rsp+680h] [rbp-100h] BYREF
  unsigned int v80; // [rsp+688h] [rbp-F8h]
  char v81; // [rsp+690h] [rbp-F0h] BYREF

  v46 = &v51;
  memset(v44, 0, sizeof(v44));
  v44[1] = (unsigned __int64)&v44[4];
  v44[12] = (unsigned __int64)&v44[14];
  v6 = *(_QWORD *)(a2 + 328);
  v7 = *(_QWORD *)(v6 + 112);
  v52 = v55;
  v8 = *(unsigned int *)(v6 + 120);
  v51 = v6;
  v55[1] = v7;
  v55[2] = v6;
  v55[0] = v7 + 8 * v8;
  LODWORD(v44[2]) = 8;
  BYTE4(v44[3]) = 1;
  HIDWORD(v44[13]) = 8;
  v47 = 8;
  v49 = 0;
  v50 = 1;
  v54 = 8;
  v48 = 1;
  v45 = 1;
  v53 = 1;
  sub_2DACB60((__int64)&v45, a2, v7, v55[0], a5, a6);
  sub_2DACDE0((__int64)v63, (__int64)v44);
  sub_2DACDE0((__int64)v56, (__int64)&v45);
  sub_2DACDE0((__int64)v70, (__int64)v56);
  sub_2DACDE0((__int64)v76, (__int64)v63);
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  if ( !v58 )
    _libc_free(v57);
  if ( v67 != v69 )
    _libc_free((unsigned __int64)v67);
  if ( !v65 )
    _libc_free(v64);
  if ( v52 != v55 )
    _libc_free((unsigned __int64)v52);
  if ( !v50 )
    _libc_free((unsigned __int64)v46);
  if ( (unsigned __int64 *)v44[12] != &v44[14] )
    _libc_free(v44[12]);
  if ( !BYTE4(v44[3]) )
    _libc_free(v44[1]);
  sub_C8CD80((__int64)v56, (__int64)v59, (__int64)v70, v9, v10, v11);
  v60 = v62;
  v61 = 0x800000000LL;
  if ( v74 )
    sub_2DACF00((__int64)&v60, (__int64 *)&v73, v12, v13, v14, v15);
  sub_C8CD80((__int64)v63, (__int64)v66, (__int64)v76, v13, v14, v15);
  v19 = v80;
  v67 = v69;
  v68 = 0x800000000LL;
  if ( v80 )
  {
    sub_2DACF00((__int64)&v67, (__int64 *)&v79, v16, v80, v17, v18);
    v19 = (unsigned int)v68;
  }
  v43 = 0;
  v20 = (unsigned int)v61;
  while ( 1 )
  {
    v21 = 24 * v20;
    if ( v20 != v19 )
      goto LABEL_26;
    if ( v60 == &v60[v21] )
      break;
    v22 = v67;
    v23 = v60;
    while ( v23[2] == v22[2] && v23[1] == v22[1] && *v23 == *v22 )
    {
      v23 += 3;
      v22 += 3;
      if ( &v60[v21] == (_BYTE *)v23 )
        goto LABEL_49;
    }
LABEL_26:
    v24 = *(_QWORD *)&v60[v21 - 8];
    v25 = v24;
    v26 = (_QWORD *)(v24 + 48);
    sub_2E225E0(a1 + 2, v24);
    v30 = (__int64 *)(*(_QWORD *)(v24 + 48) & 0xFFFFFFFFFFFFFFF8LL);
    v31 = (__int64)v30;
    if ( !v30 )
      BUG();
    v32 = (_QWORD *)(*(_QWORD *)(v24 + 48) & 0xFFFFFFFFFFFFFFF8LL);
    v33 = *v30;
    if ( (v33 & 4) == 0 && (*(_BYTE *)(v31 + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        v34 = v33 & 0xFFFFFFFFFFFFFFF8LL;
        v32 = (_QWORD *)v34;
        if ( (*(_BYTE *)(v34 + 44) & 4) == 0 )
          break;
        v33 = *(_QWORD *)v34;
      }
    }
    if ( v26 != v32 )
    {
      while ( 1 )
      {
        v35 = *v32 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v35 )
          BUG();
        v36 = *(_QWORD *)v35;
        v37 = *v32 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v35 & 4) == 0 && (*(_BYTE *)(v35 + 44) & 4) != 0 )
        {
          while ( 1 )
          {
            v38 = v36 & 0xFFFFFFFFFFFFFFF8LL;
            v37 = v38;
            if ( (*(_BYTE *)(v38 + 44) & 4) == 0 )
              break;
            v36 = *(_QWORD *)v38;
          }
        }
        v25 = *a1;
        v39 = sub_2E8B690(v32, *a1, a1 + 2);
        if ( v39 )
        {
          v43 = v39;
          sub_2E88E20(v32);
          if ( v26 == (_QWORD *)v37 )
            break;
        }
        else
        {
          v25 = (__int64)v32;
          sub_2E21F40(a1 + 2, v32);
          if ( v26 == (_QWORD *)v37 )
            break;
        }
        v32 = (_QWORD *)v37;
      }
    }
    v40 = (_DWORD)v61 == 1;
    v20 = (unsigned int)(v61 - 1);
    LODWORD(v61) = v61 - 1;
    if ( !v40 )
    {
      sub_2DACB60((__int64)v56, v25, v31, v27, v28, v29);
      v20 = (unsigned int)v61;
    }
    v19 = (unsigned int)v68;
  }
LABEL_49:
  if ( v67 != v69 )
    _libc_free((unsigned __int64)v67);
  if ( !v65 )
    _libc_free(v64);
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  if ( !v58 )
    _libc_free(v57);
  if ( v79 != &v81 )
    _libc_free((unsigned __int64)v79);
  if ( !v78 )
    _libc_free(v77);
  if ( v73 != &v75 )
    _libc_free((unsigned __int64)v73);
  if ( !v72 )
    _libc_free(v71);
  v41 = 8LL * *((unsigned int *)a1 + 8);
  if ( v41 )
    memset((void *)a1[3], 0, v41);
  return v43;
}
