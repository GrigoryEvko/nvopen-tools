// Function: sub_C31010
// Address: 0xc31010
//
void __fastcall sub_C31010(__int64 a1, int a2, __int64 a3, int a4, __int64 *a5)
{
  __int64 v6; // rsi
  __int64 v8; // rdi
  __int64 *v10; // r13
  __int64 v11; // rdx
  __int64 *v12; // rbx
  unsigned int v13; // ecx
  __int64 v14; // rdi
  __int64 *v15; // rbx
  __int64 *v16; // r12
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rdx
  int v27; // r8d
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rdx
  int v32; // edi
  __int64 v33; // rdx
  __int64 v34; // r9
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rdx
  int v40; // ecx
  __int64 v41; // rdx
  int v42; // edx
  char **v43; // rsi
  unsigned int v44; // edx
  char *v45; // rbx
  unsigned int v46; // ecx
  char *v47; // r12
  __int64 v48; // rdi
  __int64 *v49; // r14
  __int64 *v50; // r12
  __int64 v51; // rdi
  __int64 v52; // rdi
  __int64 v53; // rcx
  int v54; // edx
  __int64 v55; // [rsp+0h] [rbp-140h]
  __int64 v56; // [rsp+30h] [rbp-110h]
  __int64 v57; // [rsp+38h] [rbp-108h]
  char *v58[2]; // [rsp+40h] [rbp-100h] BYREF
  _BYTE v59[32]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 *v60; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v61; // [rsp+78h] [rbp-C8h]
  __int64 v62; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v63; // [rsp+88h] [rbp-B8h]
  __int64 v64; // [rsp+90h] [rbp-B0h]
  __int64 v65; // [rsp+98h] [rbp-A8h]
  __int64 v66; // [rsp+A0h] [rbp-A0h]
  __int64 v67; // [rsp+A8h] [rbp-98h]
  __int64 v68; // [rsp+B0h] [rbp-90h]
  char *v69; // [rsp+B8h] [rbp-88h] BYREF
  __int64 v70; // [rsp+C0h] [rbp-80h]
  _BYTE v71[32]; // [rsp+C8h] [rbp-78h] BYREF
  __int64 *v72; // [rsp+E8h] [rbp-58h] BYREF
  unsigned __int64 v73; // [rsp+F0h] [rbp-50h]
  __int64 v74; // [rsp+F8h] [rbp-48h] BYREF
  __int64 v75; // [rsp+100h] [rbp-40h]

  v6 = a3;
  v8 = a1 + 168;
  *(_QWORD *)(v8 - 152) = a3;
  *(_DWORD *)(v8 - 144) = a4;
  *(_DWORD *)(v8 - 160) = a2;
  *(_QWORD *)(v8 - 168) = &unk_49DBE40;
  *(_BYTE *)(v8 - 8) = 0;
  sub_CB1A80(v8, a3, a1, 70);
  if ( *(_BYTE *)(a1 + 160) )
  {
    if ( *((_BYTE *)a5 + 128) )
    {
      v26 = *a5;
      v27 = *((_DWORD *)a5 + 12);
      *a5 = 0;
      v64 = v26;
      v28 = a5[1];
      v69 = v71;
      v65 = v28;
      v29 = a5[2];
      a5[1] = 0;
      v66 = v29;
      v30 = a5[3];
      *((_DWORD *)a5 + 4) = 0;
      v67 = v30;
      v31 = a5[4];
      v70 = 0x400000000LL;
      v68 = v31;
      if ( v27 )
        sub_C2F210((__int64)&v69, (char **)a5 + 5);
      v32 = *((_DWORD *)a5 + 24);
      v73 = 0;
      v72 = &v74;
      if ( v32 )
        sub_C2F080((__int64 *)&v72, (__int64)(a5 + 11));
      v33 = a5[13];
      a5[4] = 0;
      v34 = a1 + 72;
      a5[3] = 0;
      v35 = v64;
      v74 = v33;
      v36 = a5[14];
      a5[13] = 0;
      *((_DWORD *)a5 + 12) = 0;
      *((_DWORD *)a5 + 24) = 0;
      v75 = v36;
      v37 = *(_QWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 32) = v35;
      v64 = v37;
      LODWORD(v37) = *(_DWORD *)(a1 + 40);
      *(_DWORD *)(a1 + 40) = v65;
      LODWORD(v65) = v37;
      LODWORD(v37) = *(_DWORD *)(a1 + 44);
      *(_DWORD *)(a1 + 44) = HIDWORD(v65);
      LODWORD(v35) = v66;
      HIDWORD(v65) = v37;
      LODWORD(v66) = *(_DWORD *)(a1 + 48);
      v38 = *(_QWORD *)(a1 + 56);
      v58[0] = v59;
      v56 = v38;
      v39 = *(_QWORD *)(a1 + 64);
      *(_DWORD *)(a1 + 48) = v35;
      v57 = v39;
      v58[1] = (char *)0x400000000LL;
      if ( *(_DWORD *)(a1 + 80) )
      {
        sub_C2F210((__int64)v58, (char **)(a1 + 72));
        v34 = a1 + 72;
      }
      v40 = *(_DWORD *)(a1 + 128);
      v61 = 0;
      v60 = &v62;
      if ( v40 )
      {
        v55 = v34;
        sub_C2F080((__int64 *)&v60, a1 + 120);
        v34 = v55;
      }
      v41 = *(_QWORD *)(a1 + 136);
      *(_DWORD *)(a1 + 80) = 0;
      *(_DWORD *)(a1 + 128) = 0;
      v62 = v41;
      v63 = *(_QWORD *)(a1 + 144);
      *(_QWORD *)(a1 + 56) = v67;
      *(_QWORD *)(a1 + 64) = v68;
      *(_QWORD *)(a1 + 136) = v74;
      *(_QWORD *)(a1 + 144) = v75;
      sub_C2F210(v34, &v69);
      if ( (__int64 **)(a1 + 120) != &v72 )
      {
        v42 = v73;
        if ( (_DWORD)v73 )
        {
          v52 = *(_QWORD *)(a1 + 120);
          if ( v52 != a1 + 136 )
          {
            _libc_free(v52, &v69);
            v42 = v73;
          }
          v53 = (__int64)v72;
          *(_DWORD *)(a1 + 128) = v42;
          v54 = HIDWORD(v73);
          HIDWORD(v73) = 0;
          *(_QWORD *)(a1 + 120) = v53;
          *(_DWORD *)(a1 + 132) = v54;
          v72 = &v74;
        }
        else
        {
          *(_DWORD *)(a1 + 128) = 0;
        }
      }
      v43 = v58;
      LODWORD(v70) = 0;
      LODWORD(v73) = 0;
      v67 = v56;
      v68 = v57;
      v74 = v62;
      v75 = v63;
      sub_C2F210((__int64)&v69, v58);
      v44 = v61;
      if ( (_DWORD)v61 )
      {
        if ( v72 != &v74 )
        {
          _libc_free(v72, v58);
          v44 = v61;
        }
        v73 = __PAIR64__(HIDWORD(v61), v44);
        v72 = v60;
      }
      else
      {
        LODWORD(v73) = 0;
        if ( v60 != &v62 )
          _libc_free(v60, v58);
      }
      if ( v58[0] != v59 )
        _libc_free(v58[0], v58);
      v45 = v69;
      LOBYTE(v46) = 0;
      v47 = &v69[8 * (unsigned int)v70];
      if ( v69 != v47 )
      {
        while ( 1 )
        {
          v48 = *(_QWORD *)v45;
          v45 += 8;
          v43 = (char **)(4096LL << v46);
          sub_C7D6A0(v48, 4096LL << v46, 16);
          if ( v47 == v45 )
            break;
          v46 = (unsigned int)((v45 - v69) >> 3) >> 7;
          if ( (unsigned int)((v45 - v69) >> 3) > 0xEFF )
            LOBYTE(v46) = 30;
        }
      }
      v49 = v72;
      v50 = &v72[2 * (unsigned int)v73];
      if ( v72 != v50 )
      {
        do
        {
          v43 = (char **)v49[1];
          v51 = *v49;
          v49 += 2;
          sub_C7D6A0(v51, v43, 16);
        }
        while ( v50 != v49 );
        v50 = v72;
      }
      if ( v50 != &v74 )
        _libc_free(v50, v43);
      if ( v69 != v71 )
        _libc_free(v69, v43);
      _libc_free(v64, v43);
      *(_QWORD *)(a1 + 152) = a5[15];
    }
    else
    {
      v10 = *(__int64 **)(a1 + 72);
      v11 = *(unsigned int *)(a1 + 80);
      *(_BYTE *)(a1 + 160) = 0;
      v12 = &v10[v11];
      if ( v10 != v12 )
      {
        LOBYTE(v13) = 0;
        while ( 1 )
        {
          v14 = *v10++;
          v6 = 4096LL << v13;
          sub_C7D6A0(v14, 4096LL << v13, 16);
          if ( v12 == v10 )
            break;
          v13 = (unsigned int)(((__int64)v10 - *(_QWORD *)(a1 + 72)) >> 3) >> 7;
          if ( (unsigned int)(((__int64)v10 - *(_QWORD *)(a1 + 72)) >> 3) > 0xEFF )
            LOBYTE(v13) = 30;
        }
      }
      v15 = *(__int64 **)(a1 + 120);
      v16 = &v15[2 * *(unsigned int *)(a1 + 128)];
      if ( v15 != v16 )
      {
        do
        {
          v6 = v15[1];
          v17 = *v15;
          v15 += 2;
          sub_C7D6A0(v17, v6, 16);
        }
        while ( v16 != v15 );
        v16 = *(__int64 **)(a1 + 120);
      }
      if ( v16 != (__int64 *)(a1 + 136) )
        _libc_free(v16, v6);
      v18 = *(_QWORD *)(a1 + 72);
      if ( v18 != a1 + 88 )
        _libc_free(v18, v6);
      _libc_free(*(_QWORD *)(a1 + 32), v6);
    }
  }
  else if ( *((_BYTE *)a5 + 128) )
  {
    v19 = *a5;
    *a5 = 0;
    *(_QWORD *)(a1 + 32) = v19;
    v20 = a5[1];
    a5[1] = 0;
    *(_QWORD *)(a1 + 40) = v20;
    v21 = a5[2];
    *((_DWORD *)a5 + 4) = 0;
    *(_QWORD *)(a1 + 48) = v21;
    *(_QWORD *)(a1 + 56) = a5[3];
    v22 = a5[4];
    *(_QWORD *)(a1 + 80) = 0x400000000LL;
    *(_QWORD *)(a1 + 64) = v22;
    *(_QWORD *)(a1 + 72) = a1 + 88;
    if ( *((_DWORD *)a5 + 12) )
      sub_C2F210(a1 + 72, (char **)a5 + 5);
    *(_QWORD *)(a1 + 128) = 0;
    *(_QWORD *)(a1 + 120) = a1 + 136;
    if ( *((_DWORD *)a5 + 24) )
      sub_C2F080((__int64 *)(a1 + 120), (__int64)(a5 + 11));
    v23 = a5[13];
    a5[4] = 0;
    a5[3] = 0;
    *(_QWORD *)(a1 + 136) = v23;
    v24 = a5[14];
    a5[13] = 0;
    *(_QWORD *)(a1 + 144) = v24;
    v25 = a5[15];
    *((_DWORD *)a5 + 12) = 0;
    *((_DWORD *)a5 + 24) = 0;
    *(_QWORD *)(a1 + 152) = v25;
    *(_BYTE *)(a1 + 160) = 1;
  }
}
