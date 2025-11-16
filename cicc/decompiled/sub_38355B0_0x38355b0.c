// Function: sub_38355B0
// Address: 0x38355b0
//
__int64 __fastcall sub_38355B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rbx
  unsigned int *v8; // r12
  __int64 v9; // rsi
  unsigned int v10; // eax
  unsigned __int64 v11; // rcx
  unsigned int *v12; // rax
  unsigned int *v13; // rdx
  unsigned int *i; // rdx
  __int64 v15; // r15
  unsigned int *v16; // rbx
  __int64 v17; // r12
  __int64 v18; // r9
  int v19; // ecx
  unsigned int v20; // edi
  __int64 v21; // rax
  int v22; // r10d
  __int64 v23; // rdx
  int v24; // ecx
  unsigned __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rax
  unsigned int *v28; // rcx
  __int64 v29; // rax
  unsigned __int16 v30; // r15
  __int64 v31; // r8
  unsigned __int16 *v32; // rax
  unsigned __int64 v33; // rdx
  unsigned __int16 *v34; // rsi
  _QWORD *v35; // r13
  unsigned int *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r9
  unsigned __int8 *v39; // r13
  unsigned __int8 *v40; // rax
  unsigned __int64 v41; // r13
  unsigned int v42; // r12d
  unsigned __int64 v43; // r15
  __int64 v44; // rdx
  int v46; // eax
  unsigned __int16 *v47; // rax
  unsigned __int16 *v48; // rdx
  _QWORD *v49; // r13
  unsigned int *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r9
  int v53; // r8d
  unsigned __int64 v54; // [rsp+8h] [rbp-1B8h]
  unsigned __int64 v55; // [rsp+20h] [rbp-1A0h]
  __int128 v56; // [rsp+30h] [rbp-190h]
  __int64 v57; // [rsp+30h] [rbp-190h]
  __int128 v58; // [rsp+30h] [rbp-190h]
  int v59; // [rsp+44h] [rbp-17Ch]
  int *v60; // [rsp+48h] [rbp-178h]
  __int64 v61; // [rsp+60h] [rbp-160h] BYREF
  unsigned int v62; // [rsp+68h] [rbp-158h]
  unsigned int *v63; // [rsp+70h] [rbp-150h] BYREF
  __int64 v64; // [rsp+78h] [rbp-148h]
  _BYTE v65[128]; // [rsp+80h] [rbp-140h] BYREF
  unsigned __int16 *v66; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v67; // [rsp+108h] [rbp-B8h]
  _BYTE v68[176]; // [rsp+110h] [rbp-B0h] BYREF

  v8 = (unsigned int *)a2;
  v9 = *(_QWORD *)(a2 + 80);
  v61 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v61, v9, 1);
  v10 = v8[18];
  v11 = v8[16];
  v64 = 0x800000000LL;
  v62 = v10;
  v12 = (unsigned int *)v65;
  v59 = v11;
  v13 = (unsigned int *)v65;
  v55 = v11;
  v63 = (unsigned int *)v65;
  if ( !v11 )
  {
    v28 = (unsigned int *)v65;
    HIDWORD(v67) = 8;
    v66 = (unsigned __int16 *)v68;
    v34 = (unsigned __int16 *)v68;
    goto LABEL_23;
  }
  if ( v11 > 8 )
  {
    sub_C8D5F0((__int64)&v63, v65, v11, 0x10u, a5, a6);
    v13 = v63;
    v12 = &v63[4 * (unsigned int)v64];
  }
  for ( i = &v13[4 * v55]; i != v12; v12 += 4 )
  {
    if ( v12 )
    {
      *(_QWORD *)v12 = 0;
      v12[2] = 0;
    }
  }
  v54 = v6;
  v15 = 0;
  v16 = v8;
  v17 = 0;
  LODWORD(v64) = v59;
  do
  {
    LODWORD(v66) = sub_375D5B0(a1, *(_QWORD *)(*((_QWORD *)v16 + 5) + v15), *(_QWORD *)(*((_QWORD *)v16 + 5) + v15 + 8));
    v60 = sub_3805BC0(a1 + 712, (int *)&v66);
    sub_37593F0(a1, v60);
    if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
    {
      v18 = a1 + 520;
      v19 = 7;
    }
    else
    {
      v26 = *(unsigned int *)(a1 + 528);
      v18 = *(_QWORD *)(a1 + 520);
      if ( !(_DWORD)v26 )
        goto LABEL_18;
      v19 = v26 - 1;
    }
    v20 = v19 & (37 * *v60);
    v21 = v18 + 24LL * v20;
    v22 = *(_DWORD *)v21;
    if ( *v60 == *(_DWORD *)v21 )
      goto LABEL_13;
    v46 = 1;
    while ( v22 != -1 )
    {
      v53 = v46 + 1;
      v20 = v19 & (v46 + v20);
      v21 = v18 + 24LL * v20;
      v22 = *(_DWORD *)v21;
      if ( *v60 == *(_DWORD *)v21 )
        goto LABEL_13;
      v46 = v53;
    }
    if ( (*(_BYTE *)(a1 + 512) & 1) != 0 )
    {
      v27 = 192;
      goto LABEL_19;
    }
    v26 = *(unsigned int *)(a1 + 528);
LABEL_18:
    v27 = 24 * v26;
LABEL_19:
    v21 = v18 + v27;
LABEL_13:
    v23 = *(_QWORD *)(v21 + 8);
    v24 = *(_DWORD *)(v21 + 16);
    v15 += 40;
    v25 = (unsigned __int64)v63;
    *(_QWORD *)&v63[v17] = v23;
    *(_DWORD *)(v25 + v17 * 4 + 8) = v24;
    v17 += 4;
  }
  while ( 40 * v55 != v15 );
  v28 = v63;
  v8 = v16;
  v6 = v54;
  v29 = *(_QWORD *)(*(_QWORD *)v63 + 48LL) + 16LL * v63[2];
  v30 = *(_WORD *)v29;
  v31 = *(_QWORD *)(v29 + 8);
  v67 = 0x800000000LL;
  v32 = (unsigned __int16 *)v68;
  v66 = (unsigned __int16 *)v68;
  v33 = v55;
  if ( v55 > 8 )
  {
    v57 = v31;
    sub_C8D5F0((__int64)&v66, v68, v55, 0x10u, v31, v18);
    v47 = v66;
    v48 = &v66[8 * v55];
    do
    {
      if ( v47 )
      {
        *v47 = v30;
        *((_QWORD *)v47 + 1) = v57;
      }
      v47 += 8;
    }
    while ( v48 != v47 );
    v49 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v58 = v63;
    *((_QWORD *)&v58 + 1) = (unsigned int)v64;
    LODWORD(v67) = v59;
    v50 = (unsigned int *)sub_33E5830(v49, v66, v55);
    v39 = sub_3411630(v49, v8[6], (__int64)&v61, v50, v51, v52, v58);
    goto LABEL_24;
  }
  do
  {
    *v32 = v30;
    v32 += 8;
    *((_QWORD *)v32 - 1) = v31;
    --v33;
  }
  while ( v33 );
  v34 = v66;
LABEL_23:
  v35 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v56 = v28;
  *((_QWORD *)&v56 + 1) = (unsigned int)v64;
  LODWORD(v67) = v59;
  v36 = (unsigned int *)sub_33E5830(v35, v34, v55);
  v39 = sub_3411630(v35, v8[6], (__int64)&v61, v36, v37, v38, v56);
  if ( v59 )
  {
LABEL_24:
    v40 = v39;
    v41 = (unsigned __int64)v8;
    v42 = 0;
    v43 = (unsigned __int64)v40;
    do
    {
      v44 = v42++;
      v6 = v44 | v6 & 0xFFFFFFFF00000000LL;
      sub_375F010(a1, v41, v44, v43, v6);
    }
    while ( v59 != v42 );
  }
  if ( v66 != (unsigned __int16 *)v68 )
    _libc_free((unsigned __int64)v66);
  if ( v63 != (unsigned int *)v65 )
    _libc_free((unsigned __int64)v63);
  if ( v61 )
    sub_B91220((__int64)&v61, v61);
  return 0;
}
