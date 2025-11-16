// Function: sub_23089D0
// Address: 0x23089d0
//
__int64 *__fastcall sub_23089D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rdi
  void *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  _QWORD *v16; // r12
  _QWORD *v17; // r14
  unsigned __int64 v18; // r15
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  bool v21; // zf
  __int64 v22; // rsi
  _QWORD *v23; // rbx
  _QWORD *v24; // r12
  unsigned __int64 v25; // r14
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  _QWORD v29[6]; // [rsp+0h] [rbp-250h] BYREF
  char v30; // [rsp+30h] [rbp-220h]
  __int64 v31; // [rsp+38h] [rbp-218h]
  __int64 v32; // [rsp+40h] [rbp-210h]
  __int64 v33; // [rsp+48h] [rbp-208h]
  unsigned int v34; // [rsp+50h] [rbp-200h]
  unsigned __int64 v35; // [rsp+58h] [rbp-1F8h]
  __int64 v36; // [rsp+60h] [rbp-1F0h]
  __int64 v37; // [rsp+68h] [rbp-1E8h]
  __int64 v38; // [rsp+70h] [rbp-1E0h]
  _QWORD *v39; // [rsp+78h] [rbp-1D8h]
  __int64 v40; // [rsp+80h] [rbp-1D0h]
  unsigned int v41; // [rsp+88h] [rbp-1C8h]
  __int64 v42; // [rsp+90h] [rbp-1C0h]
  __int64 v43; // [rsp+98h] [rbp-1B8h]
  __int64 v44; // [rsp+A0h] [rbp-1B0h]
  unsigned int v45; // [rsp+A8h] [rbp-1A8h]
  char v46[8]; // [rsp+B0h] [rbp-1A0h] BYREF
  unsigned __int64 v47; // [rsp+B8h] [rbp-198h]
  char v48; // [rsp+CCh] [rbp-184h]
  char v49[64]; // [rsp+D0h] [rbp-180h] BYREF
  __int64 v50; // [rsp+110h] [rbp-140h]
  __int64 v51; // [rsp+118h] [rbp-138h]
  __int64 v52; // [rsp+120h] [rbp-130h]
  __int64 v53; // [rsp+128h] [rbp-128h]
  __int64 v54; // [rsp+130h] [rbp-120h]
  __int64 v55; // [rsp+138h] [rbp-118h]
  char v56; // [rsp+140h] [rbp-110h]
  __int64 v57; // [rsp+148h] [rbp-108h]
  __int64 v58; // [rsp+150h] [rbp-100h]
  __int64 v59; // [rsp+158h] [rbp-F8h]
  unsigned int v60; // [rsp+160h] [rbp-F0h]
  unsigned __int64 v61; // [rsp+168h] [rbp-E8h]
  __int64 v62; // [rsp+170h] [rbp-E0h]
  __int64 v63; // [rsp+178h] [rbp-D8h]
  __int64 v64; // [rsp+180h] [rbp-D0h]
  _QWORD *v65; // [rsp+188h] [rbp-C8h]
  __int64 v66; // [rsp+190h] [rbp-C0h]
  unsigned int v67; // [rsp+198h] [rbp-B8h]
  __int64 v68; // [rsp+1A0h] [rbp-B0h]
  __int64 v69; // [rsp+1A8h] [rbp-A8h]
  __int64 v70; // [rsp+1B0h] [rbp-A0h]
  unsigned int v71; // [rsp+1B8h] [rbp-98h]
  char v72[8]; // [rsp+1C0h] [rbp-90h] BYREF
  unsigned __int64 v73; // [rsp+1C8h] [rbp-88h]
  char v74; // [rsp+1DCh] [rbp-74h]
  _BYTE v75[112]; // [rsp+1E0h] [rbp-70h] BYREF

  sub_FD5690((__int64)v29, a2 + 8, a3, a4);
  ++v31;
  v50 = v29[0];
  v57 = 1;
  v51 = v29[1];
  v52 = v29[2];
  v53 = v29[3];
  v54 = v29[4];
  v55 = v29[5];
  v56 = v30;
  v5 = v32;
  v32 = 0;
  v58 = v5;
  v6 = v33;
  v33 = 0;
  v59 = v6;
  LODWORD(v6) = v34;
  v34 = 0;
  v60 = v6;
  v61 = v35;
  v62 = v36;
  v63 = v37;
  v37 = 0;
  ++v38;
  v65 = v39;
  ++v42;
  v66 = v40;
  v36 = 0;
  v67 = v41;
  v35 = 0;
  v69 = v43;
  v64 = 1;
  v70 = v44;
  v39 = 0;
  v71 = v45;
  v40 = 0;
  v41 = 0;
  v68 = 1;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  sub_C8CF70((__int64)v72, v75, 8, (__int64)v49, (__int64)v46);
  v7 = sub_22077B0(0x118u);
  v8 = v7;
  if ( v7 )
  {
    ++v57;
    *(_QWORD *)(v7 + 64) = 1;
    v9 = v7 + 184;
    v10 = (void *)(v7 + 216);
    *(_QWORD *)v7 = &unk_4A0ACF0;
    *(_QWORD *)(v7 + 8) = v50;
    *(_QWORD *)(v7 + 16) = v51;
    *(_QWORD *)(v7 + 24) = v52;
    *(_QWORD *)(v7 + 32) = v53;
    *(_QWORD *)(v7 + 40) = v54;
    *(_QWORD *)(v7 + 48) = v55;
    *(_BYTE *)(v7 + 56) = v56;
    v11 = v58;
    v58 = 0;
    *(_QWORD *)(v8 + 72) = v11;
    v12 = v59;
    v59 = 0;
    *(_QWORD *)(v8 + 80) = v12;
    LODWORD(v12) = v60;
    v60 = 0;
    *(_DWORD *)(v8 + 88) = v12;
    *(_QWORD *)(v8 + 96) = v61;
    *(_QWORD *)(v8 + 104) = v62;
    v13 = v63;
    *(_QWORD *)(v8 + 120) = 1;
    *(_QWORD *)(v8 + 112) = v13;
    v14 = (__int64)v65;
    *(_QWORD *)(v8 + 152) = 1;
    *(_QWORD *)(v8 + 128) = v14;
    ++v64;
    *(_QWORD *)(v8 + 136) = v66;
    ++v68;
    *(_DWORD *)(v8 + 144) = v67;
    v63 = 0;
    *(_QWORD *)(v8 + 160) = v69;
    v62 = 0;
    *(_QWORD *)(v8 + 168) = v70;
    v61 = 0;
    *(_DWORD *)(v8 + 176) = v71;
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v69 = 0;
    v70 = 0;
    v71 = 0;
    sub_C8CF70(v9, v10, 8, (__int64)v75, (__int64)v72);
  }
  if ( !v74 )
    _libc_free(v73);
  sub_C7D6A0(v69, 16LL * v71, 8);
  v15 = v67;
  if ( v67 )
  {
    v16 = v65;
    v17 = &v65[2 * v67];
    do
    {
      if ( *v16 != -4096 && *v16 != -8192 )
      {
        v18 = v16[1];
        if ( v18 )
        {
          v19 = *(_QWORD *)(v18 + 96);
          if ( v19 != v18 + 112 )
            _libc_free(v19);
          v20 = *(_QWORD *)(v18 + 24);
          if ( v20 != v18 + 40 )
            _libc_free(v20);
          j_j___libc_free_0(v18);
        }
      }
      v16 += 2;
    }
    while ( v17 != v16 );
    v15 = v67;
  }
  sub_C7D6A0((__int64)v65, 16 * v15, 8);
  if ( v61 )
    j_j___libc_free_0(v61);
  sub_C7D6A0(v58, 16LL * v60, 8);
  v21 = v48 == 0;
  *a1 = v8;
  if ( v21 )
    _libc_free(v47);
  sub_C7D6A0(v43, 16LL * v45, 8);
  v22 = v41;
  if ( v41 )
  {
    v23 = v39;
    v24 = &v39[2 * v41];
    do
    {
      if ( *v23 != -8192 && *v23 != -4096 )
      {
        v25 = v23[1];
        if ( v25 )
        {
          v26 = *(_QWORD *)(v25 + 96);
          if ( v26 != v25 + 112 )
            _libc_free(v26);
          v27 = *(_QWORD *)(v25 + 24);
          if ( v27 != v25 + 40 )
            _libc_free(v27);
          j_j___libc_free_0(v25);
        }
      }
      v23 += 2;
    }
    while ( v24 != v23 );
    v22 = v41;
  }
  sub_C7D6A0((__int64)v39, 16 * v22, 8);
  if ( v35 )
    j_j___libc_free_0(v35);
  sub_C7D6A0(v32, 16LL * v34, 8);
  return a1;
}
