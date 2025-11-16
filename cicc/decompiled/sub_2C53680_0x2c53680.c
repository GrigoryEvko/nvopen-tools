// Function: sub_2C53680
// Address: 0x2c53680
//
__int64 __fastcall sub_2C53680(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v4; // r12d
  __int64 v7; // rax
  unsigned __int8 *v8; // rdi
  __int64 v9; // rdx
  unsigned __int8 *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // r12
  _QWORD **v13; // rbx
  int v14; // eax
  __int64 *v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 **v18; // r11
  __int64 v19; // r8
  __int64 v20; // r13
  unsigned int v21; // edx
  __int64 v22; // r9
  __int64 v23; // r10
  __int64 v24; // r10
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int8 **v27; // rsi
  int v28; // ecx
  _BYTE *v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rax
  int v32; // edx
  unsigned int v33; // r8d
  __int64 v34; // rax
  unsigned __int64 v35; // rax
  __int64 *v36; // rax
  unsigned __int8 *v37; // rax
  __int64 v38; // r14
  unsigned __int64 v39; // r13
  __int64 v40; // rax
  unsigned __int64 v41; // r13
  __int64 i; // rbx
  _BYTE *v43; // rax
  _BYTE *v44; // rax
  __int64 v45; // [rsp+8h] [rbp-178h]
  __int64 v46; // [rsp+10h] [rbp-170h]
  __int64 v47; // [rsp+18h] [rbp-168h]
  int v48; // [rsp+24h] [rbp-15Ch]
  const void **v49; // [rsp+28h] [rbp-158h]
  const void **v50; // [rsp+30h] [rbp-150h]
  int v51; // [rsp+38h] [rbp-148h]
  __int64 **v52; // [rsp+38h] [rbp-148h]
  __int64 v53; // [rsp+40h] [rbp-140h]
  __int64 v54; // [rsp+40h] [rbp-140h]
  unsigned int v55; // [rsp+48h] [rbp-138h]
  __int64 v56; // [rsp+68h] [rbp-118h]
  __int64 v57; // [rsp+70h] [rbp-110h]
  int v58; // [rsp+78h] [rbp-108h]
  unsigned __int64 v59; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v60; // [rsp+88h] [rbp-F8h]
  _BYTE v61[32]; // [rsp+90h] [rbp-F0h] BYREF
  __int16 v62; // [rsp+B0h] [rbp-D0h]
  unsigned __int64 v63; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v64; // [rsp+C8h] [rbp-B8h]
  _BYTE v65[32]; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v66; // [rsp+F0h] [rbp-90h]
  __int64 v67; // [rsp+F8h] [rbp-88h]
  __int16 v68; // [rsp+100h] [rbp-80h]
  __int64 v69; // [rsp+108h] [rbp-78h]
  void **v70; // [rsp+110h] [rbp-70h]
  void **v71; // [rsp+118h] [rbp-68h]
  __int64 v72; // [rsp+120h] [rbp-60h]
  int v73; // [rsp+128h] [rbp-58h]
  __int16 v74; // [rsp+12Ch] [rbp-54h]
  char v75; // [rsp+12Eh] [rbp-52h]
  __int64 v76; // [rsp+130h] [rbp-50h]
  __int64 v77; // [rsp+138h] [rbp-48h]
  void *v78; // [rsp+140h] [rbp-40h] BYREF
  void *v79; // [rsp+148h] [rbp-38h] BYREF

  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v2 = *(_QWORD *)(a2 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a2 + 80) || *(_DWORD *)(v2 + 36) != 383 )
    return 0;
  v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v8 = *(unsigned __int8 **)(a2 - 32 * v7);
  v9 = *v8;
  if ( (_BYTE)v9 == 17 )
  {
    v50 = (const void **)(v8 + 24);
  }
  else
  {
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v8 + 1) + 8LL) - 17 > 1 )
      return 0;
    if ( (unsigned __int8)v9 > 0x15u )
      return 0;
    v44 = sub_AD7630((__int64)v8, 0, v9);
    if ( !v44 || *v44 != 17 || *(_BYTE *)a2 != 85 )
      return 0;
    v50 = (const void **)(v44 + 24);
    v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  }
  v10 = *(unsigned __int8 **)(a2 + 32 * (1 - v7));
  v11 = *v10;
  if ( (_BYTE)v11 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v10 + 1) + 8LL) - 17 <= 1 && (unsigned __int8)v11 <= 0x15u )
    {
      v43 = sub_AD7630((__int64)v10, 0, v11);
      if ( v43 )
      {
        if ( *v43 == 17 )
        {
          v49 = (const void **)(v43 + 24);
          v7 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
          goto LABEL_12;
        }
      }
    }
    return 0;
  }
  v49 = (const void **)(v10 + 24);
LABEL_12:
  v12 = *(_QWORD *)(*(_QWORD *)(a2 - 32 * v7) + 8LL);
  v13 = *(_QWORD ***)(v12 + 24);
  BYTE4(v57) = *(_BYTE *)(v12 + 8) == 18;
  LODWORD(v57) = *(_DWORD *)(v12 + 32);
  v14 = sub_BCB060((__int64)v13);
  v15 = (__int64 *)sub_BCD140(*v13, 2 * v14);
  v16 = sub_BCE1B0(v15, v57);
  v55 = *(_DWORD *)(*(_QWORD *)(v12 + 24) + 8LL) >> 8;
  v17 = sub_DFD060(*(__int64 **)(a1 + 152), 49, *(_QWORD *)(a2 + 8), v16);
  v18 = *(__int64 ***)(a1 + 152);
  v19 = *(unsigned int *)(a1 + 192);
  v20 = v17;
  v4 = v21;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v22 = *(_QWORD *)(a2 - 8);
    v23 = v22 + 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  }
  else
  {
    v23 = a2;
    v22 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  }
  v24 = v23 - v22;
  v25 = v24 >> 5;
  v63 = (unsigned __int64)v65;
  v64 = 0x400000000LL;
  v26 = v24 >> 5;
  if ( (unsigned __int64)v24 > 0x80 )
  {
    v45 = v24;
    v46 = v22;
    v47 = v24 >> 5;
    v48 = v19;
    v52 = v18;
    v54 = v24 >> 5;
    sub_C8D5F0((__int64)&v63, v65, v25, 8u, v19, v22);
    v27 = (unsigned __int8 **)v63;
    v28 = v64;
    LODWORD(v25) = v54;
    v18 = v52;
    LODWORD(v19) = v48;
    v26 = v47;
    v29 = (_BYTE *)(v63 + 8LL * (unsigned int)v64);
    v22 = v46;
    v24 = v45;
  }
  else
  {
    v27 = (unsigned __int8 **)v65;
    v28 = 0;
    v29 = v65;
  }
  if ( v24 > 0 )
  {
    v30 = 0;
    do
    {
      *(_QWORD *)&v29[v30] = *(_QWORD *)(v22 + 4 * v30);
      v30 += 8;
      --v26;
    }
    while ( v26 );
    v27 = (unsigned __int8 **)v63;
    v28 = v64;
  }
  LODWORD(v64) = v25 + v28;
  v31 = sub_DFCEF0(v18, (unsigned __int8 *)a2, v27, (unsigned int)(v25 + v28), v19);
  if ( (_BYTE *)v63 != v65 )
  {
    v51 = v32;
    v53 = v31;
    _libc_free(v63);
    v32 = v51;
    v31 = v53;
  }
  if ( v4 == v32 )
    LOBYTE(v4) = v20 < v31;
  else
    LOBYTE(v4) = v32 > (int)v4;
  if ( (_BYTE)v4 )
  {
    sub_C449B0((__int64)&v59, v49, 2 * v55);
    v33 = 2 * v55;
    if ( v60 > 0x40 )
    {
      sub_C47690((__int64 *)&v59, v55);
      v33 = 2 * v55;
    }
    else
    {
      v34 = 0;
      if ( v55 != v60 )
        v34 = v59 << v55;
      v35 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v60) & v34;
      if ( !v60 )
        v35 = 0;
      v59 = v35;
    }
    sub_C449B0((__int64)&v63, v50, v33);
    if ( v60 > 0x40 )
      sub_C43BD0(&v59, (__int64 *)&v63);
    else
      v59 |= v63;
    if ( (unsigned int)v64 > 0x40 && v63 )
      j_j___libc_free_0_0(v63);
    v36 = (__int64 *)sub_B2BE50(*(_QWORD *)a1);
    v37 = (unsigned __int8 *)sub_ACCFD0(v36, (__int64)&v59);
    BYTE4(v56) = *(_BYTE *)(v16 + 8) == 18;
    v38 = a1 + 200;
    LODWORD(v56) = *(_DWORD *)(v16 + 32);
    v39 = sub_AD5E10(v56, v37);
    v40 = sub_BD5C60(a2);
    v75 = 7;
    v69 = v40;
    v70 = &v78;
    v71 = &v79;
    v74 = 512;
    v63 = (unsigned __int64)v65;
    v64 = 0x200000000LL;
    v78 = &unk_49DA100;
    v68 = 0;
    v72 = 0;
    v79 = &unk_49DA0B0;
    v73 = 0;
    v76 = 0;
    v77 = 0;
    v66 = 0;
    v67 = 0;
    sub_D5F1F0((__int64)&v63, a2);
    v62 = 257;
    v41 = sub_2C511B0((__int64 *)&v63, 0x31u, v39, *(__int64 ***)(a2 + 8), (__int64)v61, 0, v58, 0);
    sub_BD84D0(a2, v41);
    if ( *(_BYTE *)v41 > 0x1Cu )
    {
      sub_BD6B90((unsigned __int8 *)v41, (unsigned __int8 *)a2);
      for ( i = *(_QWORD *)(v41 + 16); i; i = *(_QWORD *)(i + 8) )
        sub_F15FC0(v38, *(_QWORD *)(i + 24));
      if ( *(_BYTE *)v41 > 0x1Cu )
        sub_F15FC0(v38, v41);
    }
    if ( *(_BYTE *)a2 > 0x1Cu )
      sub_F15FC0(v38, a2);
    nullsub_61();
    v78 = &unk_49DA100;
    nullsub_63();
    if ( (_BYTE *)v63 != v65 )
      _libc_free(v63);
    if ( v60 > 0x40 && v59 )
      j_j___libc_free_0_0(v59);
  }
  return v4;
}
