// Function: sub_28B7870
// Address: 0x28b7870
//
void __fastcall sub_28B7870(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rcx
  __int64 v6; // rax
  __int64 *v7; // rax
  int v8; // r11d
  unsigned int v9; // r12d
  unsigned int v10; // eax
  unsigned int v11; // r12d
  int v12; // eax
  int v13; // edx
  int v14; // esi
  __int64 *v15; // rsi
  __int64 v16; // r9
  __int64 v17; // rax
  int v18; // r8d
  char v19; // r10
  int v20; // r9d
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v23; // r8d
  unsigned __int64 v24; // rcx
  int v25; // r15d
  __int64 v26; // r14
  __int64 *v27; // rdx
  char v28; // r10
  int v29; // r9d
  unsigned int v30; // r8d
  __int64 *v31; // rax
  unsigned __int64 v32; // rcx
  int v33; // edi
  int v34; // eax
  __int64 *v35; // rax
  __int64 *v36; // rsi
  char v37; // bl
  __int64 v38; // r8
  __int64 v39; // rdx
  int v40; // edi
  __int64 *v41; // rax
  __int64 v42; // rsi
  __int64 *v43; // rax
  __int64 *v44; // rdx
  __int64 v45; // rsi
  unsigned int v46; // eax
  __int64 v47; // rcx
  unsigned int v48; // edx
  unsigned int v49; // eax
  __int64 v50; // rsi
  int v51; // [rsp+Ch] [rbp-204h]
  char v52; // [rsp+13h] [rbp-1FDh]
  int v53; // [rsp+14h] [rbp-1FCh]
  unsigned __int64 v54; // [rsp+18h] [rbp-1F8h]
  unsigned int v56; // [rsp+28h] [rbp-1E8h]
  int v57; // [rsp+2Ch] [rbp-1E4h]
  __int64 v58; // [rsp+30h] [rbp-1E0h]
  __int64 v59; // [rsp+38h] [rbp-1D8h]
  unsigned __int64 v60; // [rsp+40h] [rbp-1D0h]
  unsigned int v61; // [rsp+48h] [rbp-1C8h]
  int v62; // [rsp+4Ch] [rbp-1C4h]
  __int64 v63; // [rsp+50h] [rbp-1C0h]
  __int64 v64; // [rsp+58h] [rbp-1B8h]
  __int64 v65; // [rsp+60h] [rbp-1B0h]
  __int64 v66; // [rsp+68h] [rbp-1A8h] BYREF
  __int64 v67; // [rsp+70h] [rbp-1A0h]
  __int64 v68; // [rsp+78h] [rbp-198h] BYREF
  unsigned int v69; // [rsp+80h] [rbp-190h]
  _BYTE v70[4]; // [rsp+B8h] [rbp-158h] BYREF
  int v71; // [rsp+BCh] [rbp-154h]
  __int64 v72; // [rsp+C0h] [rbp-150h]
  __int64 v73; // [rsp+C8h] [rbp-148h]
  int v74; // [rsp+D0h] [rbp-140h]
  unsigned __int64 v75; // [rsp+D8h] [rbp-138h]
  unsigned int v76; // [rsp+E0h] [rbp-130h]
  __int64 v77; // [rsp+E8h] [rbp-128h]
  __int64 v78; // [rsp+F0h] [rbp-120h]
  int v79; // [rsp+F8h] [rbp-118h]
  unsigned __int64 v80; // [rsp+100h] [rbp-110h]
  unsigned int v81; // [rsp+108h] [rbp-108h]
  int v82; // [rsp+110h] [rbp-100h]
  __int64 v83; // [rsp+118h] [rbp-F8h]
  __int64 v84; // [rsp+120h] [rbp-F0h] BYREF
  __int64 v85; // [rsp+128h] [rbp-E8h] BYREF
  __int64 v86; // [rsp+130h] [rbp-E0h]
  __int64 v87; // [rsp+138h] [rbp-D8h] BYREF
  unsigned int v88; // [rsp+140h] [rbp-D0h]
  char v89; // [rsp+178h] [rbp-98h] BYREF
  int v90; // [rsp+17Ch] [rbp-94h]
  __int64 v91; // [rsp+180h] [rbp-90h]
  __int64 v92; // [rsp+188h] [rbp-88h]
  int v93; // [rsp+190h] [rbp-80h]
  unsigned __int64 v94; // [rsp+198h] [rbp-78h]
  unsigned int v95; // [rsp+1A0h] [rbp-70h]
  __int64 v96; // [rsp+1A8h] [rbp-68h]
  __int64 v97; // [rsp+1B0h] [rbp-60h]
  int v98; // [rsp+1B8h] [rbp-58h]
  unsigned __int64 v99; // [rsp+1C0h] [rbp-50h]
  unsigned int v100; // [rsp+1C8h] [rbp-48h]
  int v101; // [rsp+1D0h] [rbp-40h]
  __int64 v102; // [rsp+1D8h] [rbp-38h]

  v3 = (__int64 *)(a3 + 8);
  v6 = *(_QWORD *)a3;
  v66 = 0;
  v65 = v6;
  v7 = &v68;
  v67 = 1;
  do
    *v7++ = -4096;
  while ( v7 != (__int64 *)v70 );
  v8 = *(_DWORD *)(a3 + 20);
  v9 = *(_DWORD *)(a3 + 16);
  v10 = v9 & 0xFFFFFFFE;
  v11 = v9 >> 1;
  v12 = v67 & 1 | v10;
  v13 = HIDWORD(v67);
  v14 = v67 & 0xFFFFFFFE | *(_DWORD *)(a3 + 16) & 1;
  HIDWORD(v67) = *(_DWORD *)(a3 + 20);
  *(_DWORD *)(a3 + 16) = v14;
  LODWORD(v67) = v12;
  *(_DWORD *)(a3 + 20) = v13;
  if ( (v12 & 1) != 0 )
  {
    v15 = &v66;
    if ( (*(_BYTE *)(a3 + 16) & 1) != 0 )
    {
      v43 = &v68;
      v44 = (__int64 *)(a3 + 24);
      do
      {
        v45 = *v43;
        *v43++ = *v44;
        *v44++ = v45;
      }
      while ( v43 != (__int64 *)v70 );
      goto LABEL_8;
    }
  }
  else
  {
    v15 = v3;
    v3 = &v66;
    if ( (*(_BYTE *)(a3 + 16) & 1) == 0 )
    {
      v46 = v69;
      v47 = *(_QWORD *)(a3 + 24);
      *(_QWORD *)(a3 + 24) = v68;
      v48 = *(_DWORD *)(a3 + 32);
      v68 = v47;
      v69 = v48;
      *(_DWORD *)(a3 + 32) = v46;
      goto LABEL_9;
    }
  }
  *((_BYTE *)v3 + 8) |= 1u;
  v16 = v3[2];
  v17 = 2;
  v18 = *((_DWORD *)v3 + 6);
  do
  {
    v3[v17] = v15[v17];
    ++v17;
  }
  while ( v17 != 10 );
  *((_BYTE *)v15 + 8) &= ~1u;
  v15[2] = v16;
  *((_DWORD *)v15 + 6) = v18;
LABEL_8:
  v8 = HIDWORD(v67);
  v11 = (unsigned int)v67 >> 1;
LABEL_9:
  v19 = *(_BYTE *)(a3 + 88);
  v51 = v8;
  v20 = *(_DWORD *)(a3 + 92);
  v64 = *(_QWORD *)(a3 + 96);
  v72 = v64;
  v21 = *(_QWORD *)(a3 + 104);
  v70[0] = v19;
  v63 = v21;
  v73 = v21;
  v52 = v19;
  v62 = *(_DWORD *)(a3 + 112);
  v74 = v62;
  LODWORD(v21) = *(_DWORD *)(a3 + 128);
  v71 = v20;
  v61 = v21;
  v76 = v21;
  v53 = v20;
  v60 = *(_QWORD *)(a3 + 120);
  v75 = v60;
  v22 = *(_QWORD *)(a3 + 136);
  *(_DWORD *)(a3 + 128) = 0;
  v59 = v22;
  v77 = v22;
  v58 = *(_QWORD *)(a3 + 144);
  v78 = v58;
  v57 = *(_DWORD *)(a3 + 152);
  v79 = v57;
  v23 = *(_DWORD *)(a3 + 168);
  v24 = *(_QWORD *)(a3 + 160);
  v25 = *(_DWORD *)(a3 + 176);
  *(_DWORD *)(a3 + 168) = 0;
  v26 = *(_QWORD *)(a3 + 184);
  v81 = v23;
  v56 = v23;
  v80 = v24;
  v54 = v24;
  v82 = v25;
  v83 = v26;
  sub_28B56D0(a3, a1);
  v85 = 0;
  v27 = &v87;
  v28 = v52;
  v86 = 1;
  v29 = v53;
  v30 = v56;
  v84 = v65;
  v31 = &v87;
  v32 = v54;
  do
    *v31++ = -4096;
  while ( v31 != (__int64 *)&v89 );
  v33 = v86 & 0xFFFFFFFE;
  v34 = HIDWORD(v86);
  HIDWORD(v86) = v51;
  LODWORD(v86) = v86 & 1 | (2 * v11);
  LODWORD(v67) = v33 | v67 & 1;
  HIDWORD(v67) = v34;
  if ( (v86 & 1) != 0 )
  {
    v35 = &v85;
    v36 = &v66;
    if ( (v67 & 1) != 0 )
    {
      v41 = &v68;
      do
      {
        v42 = *v27;
        *v27++ = *v41++;
        *(v41 - 1) = v42;
      }
      while ( v70 != (_BYTE *)v41 );
      goto LABEL_16;
    }
  }
  else
  {
    v35 = &v66;
    v36 = &v85;
    v37 = v67 & 1;
    if ( (v67 & 1) == 0 )
    {
      v49 = v88;
      v50 = v68;
      v68 = v87;
      v87 = v50;
      v88 = v69;
      v69 = v49;
      goto LABEL_17;
    }
  }
  *((_BYTE *)v36 + 8) |= 1u;
  v38 = v36[2];
  v39 = 2;
  v40 = *((_DWORD *)v36 + 6);
  do
  {
    v36[v39] = v35[v39];
    ++v39;
  }
  while ( v39 != 10 );
  *((_BYTE *)v35 + 8) &= ~1u;
  v35[2] = v38;
  *((_DWORD *)v35 + 6) = v40;
LABEL_16:
  v28 = v70[0];
  v29 = v71;
  v64 = v72;
  v37 = v67 & 1;
  v30 = v81;
  v32 = v80;
  v63 = v73;
  v25 = v82;
  v26 = v83;
  v62 = v74;
  v61 = v76;
  v60 = v75;
  v59 = v77;
  v58 = v78;
  v57 = v79;
LABEL_17:
  v99 = v32;
  v100 = v30;
  v91 = v64;
  v89 = v28;
  v92 = v63;
  v90 = v29;
  v93 = v62;
  v101 = v25;
  v95 = v61;
  v102 = v26;
  v94 = v60;
  v96 = v59;
  v97 = v58;
  v98 = v57;
  sub_28B6270(a1, 0, 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 6), &v84);
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  if ( v95 > 0x40 && v94 )
    j_j___libc_free_0_0(v94);
  if ( (v86 & 1) == 0 )
    sub_C7D6A0(v87, 8LL * v88, 8);
  if ( !v37 )
    sub_C7D6A0(v68, 8LL * v69, 8);
}
