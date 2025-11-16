// Function: sub_13BA9D0
// Address: 0x13ba9d0
//
void __fastcall sub_13BA9D0(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  _BYTE *v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  char *v14; // rdi
  __int64 v15; // rdx
  _BYTE *v16; // rsi
  _BYTE *v17; // r8
  unsigned __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rdx
  _BYTE *v22; // rax
  char v23; // cl
  __int64 v24; // rcx
  __int64 v25; // r8
  unsigned __int64 v26; // r13
  __int64 v27; // rax
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdx
  __int64 v30; // rax
  char v31; // si
  unsigned __int64 v32; // rax
  __int64 v33; // rcx
  unsigned __int64 v34; // rax
  char v35; // si
  char v36; // r8
  _QWORD v37[16]; // [rsp+10h] [rbp-330h] BYREF
  __int64 v38; // [rsp+90h] [rbp-2B0h] BYREF
  _QWORD *v39; // [rsp+98h] [rbp-2A8h]
  _QWORD *v40; // [rsp+A0h] [rbp-2A0h]
  __int64 v41; // [rsp+A8h] [rbp-298h]
  int v42; // [rsp+B0h] [rbp-290h]
  _QWORD v43[8]; // [rsp+B8h] [rbp-288h] BYREF
  unsigned __int64 v44; // [rsp+F8h] [rbp-248h] BYREF
  unsigned __int64 v45; // [rsp+100h] [rbp-240h]
  unsigned __int64 v46; // [rsp+108h] [rbp-238h]
  char v47[8]; // [rsp+110h] [rbp-230h] BYREF
  __int64 v48; // [rsp+118h] [rbp-228h]
  unsigned __int64 v49; // [rsp+120h] [rbp-220h]
  _BYTE v50[64]; // [rsp+138h] [rbp-208h] BYREF
  __int64 v51; // [rsp+178h] [rbp-1C8h]
  __int64 v52; // [rsp+180h] [rbp-1C0h]
  unsigned __int64 v53; // [rsp+188h] [rbp-1B8h]
  char v54[8]; // [rsp+190h] [rbp-1B0h] BYREF
  __int64 v55; // [rsp+198h] [rbp-1A8h]
  unsigned __int64 v56; // [rsp+1A0h] [rbp-1A0h]
  _BYTE v57[64]; // [rsp+1B8h] [rbp-188h] BYREF
  unsigned __int64 v58; // [rsp+1F8h] [rbp-148h]
  unsigned __int64 i; // [rsp+200h] [rbp-140h]
  unsigned __int64 v60; // [rsp+208h] [rbp-138h]
  _QWORD v61[2]; // [rsp+210h] [rbp-130h] BYREF
  unsigned __int64 v62; // [rsp+220h] [rbp-120h]
  char v63[64]; // [rsp+238h] [rbp-108h] BYREF
  _BYTE *v64; // [rsp+278h] [rbp-C8h]
  _BYTE *v65; // [rsp+280h] [rbp-C0h]
  unsigned __int64 v66; // [rsp+288h] [rbp-B8h]
  char v67[8]; // [rsp+290h] [rbp-B0h] BYREF
  __int64 v68; // [rsp+298h] [rbp-A8h]
  unsigned __int64 v69; // [rsp+2A0h] [rbp-A0h]
  char v70[64]; // [rsp+2B8h] [rbp-88h] BYREF
  __int64 v71; // [rsp+2F8h] [rbp-48h]
  __int64 v72; // [rsp+300h] [rbp-40h]
  unsigned __int64 v73; // [rsp+308h] [rbp-38h]

  v2 = **(_QWORD **)(a1 + 8);
  v42 = 0;
  memset(v37, 0, sizeof(v37));
  v37[1] = &v37[5];
  v37[2] = &v37[5];
  v3 = *(_QWORD *)(v2 + 56);
  v39 = v43;
  v43[0] = v3;
  v61[0] = v3;
  v40 = v43;
  v41 = 0x100000008LL;
  LODWORD(v37[3]) = 8;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v38 = 1;
  LOBYTE(v62) = 0;
  sub_13B8390(&v44, (__int64)v61);
  sub_16CCEE0(v47, v50, 8, v37);
  v4 = v37[13];
  memset(&v37[13], 0, 24);
  v51 = v4;
  v52 = v37[14];
  v53 = v37[15];
  sub_16CCEE0(v54, v57, 8, &v38);
  v5 = v44;
  v44 = 0;
  v58 = v5;
  v6 = v45;
  v45 = 0;
  i = v6;
  v7 = v46;
  v46 = 0;
  v60 = v7;
  sub_16CCEE0(v61, v63, 8, v54);
  v8 = v58;
  v58 = 0;
  v64 = (_BYTE *)v8;
  v9 = (_BYTE *)i;
  i = 0;
  v65 = v9;
  v10 = v60;
  v60 = 0;
  v66 = v10;
  sub_16CCEE0(v67, v70, 8, v47);
  v11 = v51;
  v51 = 0;
  v71 = v11;
  v12 = v52;
  v52 = 0;
  v72 = v12;
  v13 = v53;
  v53 = 0;
  v73 = v13;
  if ( v58 )
    j_j___libc_free_0(v58, v60 - v58);
  if ( v56 != v55 )
    _libc_free(v56);
  if ( v51 )
    j_j___libc_free_0(v51, v53 - v51);
  if ( v49 != v48 )
    _libc_free(v49);
  if ( v44 )
    j_j___libc_free_0(v44, v46 - v44);
  if ( v40 != v39 )
    _libc_free((unsigned __int64)v40);
  if ( v37[13] )
    j_j___libc_free_0(v37[13], v37[15] - v37[13]);
  if ( v37[2] != v37[1] )
    _libc_free(v37[2]);
  v14 = v47;
  sub_16CCCB0(v47, v50, v61);
  v16 = v65;
  v17 = v64;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v18 = v65 - v64;
  if ( v65 == v64 )
  {
    v20 = 0;
  }
  else
  {
    if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_70;
    v19 = sub_22077B0(v65 - v64);
    v16 = v65;
    v17 = v64;
    v20 = v19;
  }
  v51 = v20;
  v52 = v20;
  v53 = v20 + v18;
  if ( v17 != v16 )
  {
    v21 = v20;
    v22 = v17;
    do
    {
      if ( v21 )
      {
        *(_QWORD *)v21 = *(_QWORD *)v22;
        v23 = v22[16];
        *(_BYTE *)(v21 + 16) = v23;
        if ( v23 )
          *(_QWORD *)(v21 + 8) = *((_QWORD *)v22 + 1);
      }
      v22 += 24;
      v21 += 24;
    }
    while ( v22 != v16 );
    v20 += 8 * ((unsigned __int64)(v22 - 24 - v17) >> 3) + 24;
  }
  v16 = v57;
  v52 = v20;
  v14 = v54;
  sub_16CCCB0(v54, v57, v67);
  v24 = v72;
  v25 = v71;
  v58 = 0;
  i = 0;
  v60 = 0;
  v26 = v72 - v71;
  if ( v72 == v71 )
  {
    v28 = 0;
    goto LABEL_30;
  }
  if ( v26 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_70:
    sub_4261EA(v14, v16, v15);
  v27 = sub_22077B0(v72 - v71);
  v24 = v72;
  v25 = v71;
  v28 = v27;
LABEL_30:
  v58 = v28;
  i = v28;
  v60 = v28 + v26;
  if ( v24 == v25 )
  {
    v32 = v28;
  }
  else
  {
    v29 = v28;
    v30 = v25;
    do
    {
      if ( v29 )
      {
        *(_QWORD *)v29 = *(_QWORD *)v30;
        v31 = *(_BYTE *)(v30 + 16);
        *(_BYTE *)(v29 + 16) = v31;
        if ( v31 )
          *(_QWORD *)(v29 + 8) = *(_QWORD *)(v30 + 8);
      }
      v30 += 24;
      v29 += 24LL;
    }
    while ( v24 != v30 );
    v32 = v28 + 8 * ((unsigned __int64)(v24 - 24 - v25) >> 3) + 24;
  }
  for ( i = v32; ; v32 = i )
  {
    v33 = v51;
    if ( v52 - v51 != v32 - v28 )
      goto LABEL_38;
    if ( v51 == v52 )
      break;
    v34 = v28;
    while ( *(_QWORD *)v33 == *(_QWORD *)v34 )
    {
      v35 = *(_BYTE *)(v33 + 16);
      v36 = *(_BYTE *)(v34 + 16);
      if ( v35 && v36 )
      {
        if ( *(_QWORD *)(v33 + 8) != *(_QWORD *)(v34 + 8) )
          break;
        v33 += 24;
        v34 += 24LL;
        if ( v52 == v33 )
          goto LABEL_47;
      }
      else
      {
        if ( v36 != v35 )
          break;
        v33 += 24;
        v34 += 24LL;
        if ( v52 == v33 )
          goto LABEL_47;
      }
    }
LABEL_38:
    sub_13B9900((_BYTE *)a1, *(__int64 **)(v52 - 24));
    sub_13BA8B0((__int64)v47);
    v28 = v58;
  }
LABEL_47:
  if ( v28 )
    j_j___libc_free_0(v28, v60 - v28);
  if ( v56 != v55 )
    _libc_free(v56);
  if ( v51 )
    j_j___libc_free_0(v51, v53 - v51);
  if ( v49 != v48 )
    _libc_free(v49);
  if ( v71 )
    j_j___libc_free_0(v71, v73 - v71);
  if ( v69 != v68 )
    _libc_free(v69);
  if ( v64 )
    j_j___libc_free_0(v64, v66 - (_QWORD)v64);
  if ( v62 != v61[1] )
    _libc_free(v62);
}
