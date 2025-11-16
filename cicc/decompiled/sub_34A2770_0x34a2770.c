// Function: sub_34A2770
// Address: 0x34a2770
//
__int64 __fastcall sub_34A2770(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // r13d
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v14; // r13d
  __int64 v15; // rsi
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned int v22; // r12d
  __int64 v24; // rax
  _BYTE *v25; // r14
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  int v32; // ecx
  unsigned int v33; // esi
  unsigned int v34; // esi
  __int64 v35; // rcx
  unsigned int v36; // eax
  __int64 j; // rax
  unsigned __int64 *v38; // rax
  __int64 v39; // rsi
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // r14
  __int64 v42; // r15
  __int64 i; // rax
  unsigned __int64 *v44; // rax
  __int64 v45; // rsi
  unsigned __int64 v46; // rdx
  unsigned __int64 v47; // r14
  __int64 v48; // r15
  __int64 v50; // [rsp+40h] [rbp-1C0h]
  __int64 v51; // [rsp+48h] [rbp-1B8h]
  __int64 v52; // [rsp+50h] [rbp-1B0h] BYREF
  _BYTE *v53; // [rsp+58h] [rbp-1A8h] BYREF
  __int64 v54; // [rsp+60h] [rbp-1A0h]
  _BYTE v55[72]; // [rsp+68h] [rbp-198h] BYREF
  __int64 v56; // [rsp+B0h] [rbp-150h] BYREF
  _BYTE *v57; // [rsp+B8h] [rbp-148h] BYREF
  __int64 v58; // [rsp+C0h] [rbp-140h]
  _BYTE v59[72]; // [rsp+C8h] [rbp-138h] BYREF
  __int64 v60; // [rsp+110h] [rbp-F0h] BYREF
  __int64 *v61; // [rsp+118h] [rbp-E8h]
  __int64 v62; // [rsp+120h] [rbp-E0h]
  __int64 v63; // [rsp+128h] [rbp-D8h] BYREF
  __int64 v64; // [rsp+130h] [rbp-D0h]
  __int64 v65; // [rsp+170h] [rbp-90h] BYREF
  _BYTE *v66; // [rsp+178h] [rbp-88h]
  __int64 v67; // [rsp+180h] [rbp-80h]
  _BYTE v68[120]; // [rsp+188h] [rbp-78h] BYREF

  v53 = v55;
  v54 = 0x400000000LL;
  v52 = a1 + 8;
  sub_34A26E0((__int64)&v52, 0, a3, a4, a5, a6);
  v10 = *(_DWORD *)(v52 + 192);
  if ( v10 )
  {
    v6 = (unsigned int)v54;
    for ( i = (unsigned int)(v54 - 1); v10 > (unsigned int)i; LODWORD(v54) = v54 + 1 )
    {
      v45 = (__int64)v53;
      v46 = v6 + 1;
      v7 = *(_QWORD *)(*(_QWORD *)&v53[16 * i] + 8LL * *(unsigned int *)&v53[16 * i + 12]) & 0x3FLL;
      v47 = *(_QWORD *)(*(_QWORD *)&v53[16 * i] + 8LL * *(unsigned int *)&v53[16 * i + 12]) & 0xFFFFFFFFFFFFFFC0LL;
      v48 = v7 + 1;
      if ( v46 > HIDWORD(v54) )
      {
        sub_C8D5F0((__int64)&v53, v55, v46, 0x10u, v8, v9);
        v45 = (__int64)v53;
      }
      v44 = (unsigned __int64 *)(v45 + 16LL * (unsigned int)v54);
      *v44 = v47;
      v44[1] = v48;
      i = (unsigned int)v54;
      v6 = (unsigned int)(v54 + 1);
    }
  }
  v56 = a2 + 8;
  v57 = v59;
  v58 = 0x400000000LL;
  sub_34A26E0((__int64)&v56, 0, v6, v7, v8, v9);
  v14 = *(_DWORD *)(v56 + 192);
  if ( v14 )
  {
    v11 = (unsigned int)v58;
    for ( j = (unsigned int)(v58 - 1); v14 > (unsigned int)j; LODWORD(v58) = v58 + 1 )
    {
      v39 = (__int64)v57;
      v40 = v11 + 1;
      v41 = *(_QWORD *)(*(_QWORD *)&v57[16 * j] + 8LL * *(unsigned int *)&v57[16 * j + 12]) & 0xFFFFFFFFFFFFFFC0LL;
      v42 = (*(_QWORD *)(*(_QWORD *)&v57[16 * j] + 8LL * *(unsigned int *)&v57[16 * j + 12]) & 0x3FLL) + 1;
      if ( v40 > HIDWORD(v58) )
      {
        sub_C8D5F0((__int64)&v57, v59, v40, 0x10u, v12, v13);
        v39 = (__int64)v57;
      }
      v38 = (unsigned __int64 *)(v39 + 16LL * (unsigned int)v58);
      *v38 = v41;
      v38[1] = v42;
      j = (unsigned int)v58;
      v11 = (unsigned int)(v58 + 1);
    }
  }
  while ( 1 )
  {
    v15 = *(unsigned int *)(a1 + 204);
    v60 = a1 + 8;
    v61 = &v63;
    v62 = 0x400000000LL;
    v16 = *(unsigned int *)(a1 + 200);
    if ( (_DWORD)v16 )
    {
      v11 = a1 + 16;
      v64 = (v15 << 32) | (unsigned int)v15;
      LODWORD(v62) = 1;
      v63 = a1 + 16;
    }
    else
    {
      v63 = a1 + 8;
      v64 = (v15 << 32) | (unsigned int)v15;
      LODWORD(v62) = 1;
    }
    if ( (_DWORD)v54 && (v16 = (__int64)v53, *((_DWORD *)v53 + 3) < *((_DWORD *)v53 + 2)) )
    {
      v11 = HIDWORD(v64);
      v24 = (__int64)&v53[16 * (unsigned int)v54 - 16];
      if ( *(_DWORD *)(v24 + 12) == HIDWORD(v64) )
      {
        v11 = v63;
        if ( *(_QWORD *)v24 == v63 )
          goto LABEL_7;
      }
    }
    else if ( HIDWORD(v64) >= (unsigned int)v64 )
    {
      goto LABEL_7;
    }
    v67 = 0x400000000LL;
    v65 = a2 + 8;
    v66 = v68;
    sub_34A26E0((__int64)&v65, *(_DWORD *)(a2 + 204), v11, v16, v12, v13);
    v25 = v66;
    if ( !(_DWORD)v58 )
      break;
    v16 = (__int64)v57;
    v11 = *((unsigned int *)v57 + 2);
    if ( *((_DWORD *)v57 + 3) >= (unsigned int)v11 )
      break;
    v26 = (__int64)&v57[16 * (unsigned int)v58 - 16];
    v27 = *(unsigned int *)(v26 + 12);
    v28 = *(_QWORD *)v26;
    v16 = (__int64)&v66[16 * (unsigned int)v67 - 16];
    if ( (_DWORD)v27 == *(_DWORD *)(v16 + 12) && *(_QWORD *)v16 == v28 )
      goto LABEL_22;
LABEL_21:
    v16 = *(_QWORD *)&v53[16 * (unsigned int)v54 - 16] + 16LL * *(unsigned int *)&v53[16 * (unsigned int)v54 - 4];
    v11 = *(_QWORD *)v16;
    if ( *(_QWORD *)(v28 + 16 * v27) != *(_QWORD *)v16 )
      goto LABEL_22;
    v50 = *(_QWORD *)sub_34A25B0((__int64)&v52);
    v29 = *(_QWORD *)sub_34A25B0((__int64)&v56);
    if ( v25 != v68 )
    {
      v51 = v29;
      _libc_free((unsigned __int64)v25);
      v29 = v51;
    }
    if ( v50 != v29 )
      goto LABEL_24;
    v30 = (__int64)&v53[16 * (unsigned int)v54 - 16];
    v11 = *(unsigned int *)(v30 + 12);
    *(_DWORD *)(v30 + 12) = v11 + 1;
    if ( (_DWORD)v11 + 1 == *(_DWORD *)&v53[16 * (unsigned int)v54 - 8] )
    {
      v34 = *(_DWORD *)(v52 + 192);
      if ( v34 )
        sub_F03D40((__int64 *)&v53, v34);
    }
    v31 = (__int64)&v57[16 * (unsigned int)v58 - 16];
    v32 = *(_DWORD *)(v31 + 12) + 1;
    *(_DWORD *)(v31 + 12) = v32;
    if ( v32 == *(_DWORD *)&v57[16 * (unsigned int)v58 - 8] )
    {
      v33 = *(_DWORD *)(v56 + 192);
      if ( v33 )
        sub_F03D40((__int64 *)&v57, v33);
    }
  }
  v11 = (unsigned int)v67;
  if ( (_DWORD)v67 )
  {
    v11 = *((unsigned int *)v66 + 2);
    if ( *((_DWORD *)v66 + 3) < (unsigned int)v11 )
    {
      v35 = (__int64)&v57[16 * (unsigned int)v58 - 16];
      v27 = *(unsigned int *)(v35 + 12);
      v28 = *(_QWORD *)v35;
      goto LABEL_21;
    }
  }
LABEL_22:
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
LABEL_24:
  LODWORD(v15) = *(_DWORD *)(a1 + 204);
LABEL_7:
  v60 = a1 + 8;
  v61 = &v63;
  v62 = 0x400000000LL;
  sub_34A26E0((__int64)&v60, v15, v11, v16, v12, v13);
  LOBYTE(v17) = sub_34A1E40((__int64)&v52, (__int64)&v60);
  v22 = v17;
  if ( (_BYTE)v17 )
  {
    v67 = 0x400000000LL;
    v66 = v68;
    v65 = a2 + 8;
    sub_34A26E0((__int64)&v65, *(_DWORD *)(a2 + 204), v18, v19, v20, v21);
    LOBYTE(v36) = sub_34A1E40((__int64)&v56, (__int64)&v65);
    v22 = v36;
    if ( v66 != v68 )
      _libc_free((unsigned __int64)v66);
  }
  if ( v61 != &v63 )
    _libc_free((unsigned __int64)v61);
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  return v22;
}
