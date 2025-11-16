// Function: sub_28B6C00
// Address: 0x28b6c00
//
void __fastcall sub_28B6C00(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v3; // r15
  unsigned __int64 i; // r13
  __int64 v5; // rax
  __int64 *v6; // rdx
  unsigned __int64 v7; // rdi
  __int64 *v8; // rax
  int v9; // ecx
  int v10; // esi
  int v11; // edx
  int v12; // r14d
  __int64 *v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rdx
  int v16; // edi
  __int64 v17; // r11
  __int64 v18; // r10
  int v19; // r9d
  unsigned int v20; // r8d
  int v21; // eax
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  int v26; // esi
  __int64 v27; // rcx
  __int64 *v28; // rax
  int v29; // edx
  __int64 *v30; // rax
  __int64 *v31; // rdx
  char v32; // r14
  __int64 v33; // r8
  __int64 v34; // rcx
  int v35; // edi
  __int64 *v36; // rdx
  __int64 *v37; // rax
  __int64 v38; // rsi
  __int64 *v39; // rax
  __int64 *v40; // rdx
  __int64 v41; // rsi
  unsigned int v42; // eax
  __int64 v43; // rcx
  unsigned int v44; // edx
  unsigned int v45; // eax
  __int64 v46; // [rsp+8h] [rbp-208h]
  unsigned __int64 v48; // [rsp+30h] [rbp-1E0h]
  unsigned int v49; // [rsp+38h] [rbp-1D8h]
  int v50; // [rsp+3Ch] [rbp-1D4h]
  __int64 v51; // [rsp+40h] [rbp-1D0h]
  char v52; // [rsp+4Bh] [rbp-1C5h]
  int v53; // [rsp+4Ch] [rbp-1C4h]
  int v54; // [rsp+50h] [rbp-1C0h]
  __int64 v55; // [rsp+50h] [rbp-1C0h]
  __int64 v56; // [rsp+58h] [rbp-1B8h]
  __int64 v57; // [rsp+60h] [rbp-1B0h]
  __int64 v58; // [rsp+68h] [rbp-1A8h] BYREF
  __int64 v59; // [rsp+70h] [rbp-1A0h]
  __int64 v60; // [rsp+78h] [rbp-198h] BYREF
  unsigned int v61; // [rsp+80h] [rbp-190h]
  _BYTE v62[4]; // [rsp+B8h] [rbp-158h] BYREF
  int v63; // [rsp+BCh] [rbp-154h]
  __int64 v64; // [rsp+C0h] [rbp-150h]
  __int64 v65; // [rsp+C8h] [rbp-148h]
  int v66; // [rsp+D0h] [rbp-140h]
  unsigned __int64 v67; // [rsp+D8h] [rbp-138h]
  unsigned int v68; // [rsp+E0h] [rbp-130h]
  __int64 v69; // [rsp+E8h] [rbp-128h]
  __int64 v70; // [rsp+F0h] [rbp-120h]
  int v71; // [rsp+F8h] [rbp-118h]
  unsigned __int64 v72; // [rsp+100h] [rbp-110h]
  unsigned int v73; // [rsp+108h] [rbp-108h]
  int v74; // [rsp+110h] [rbp-100h]
  __int64 v75; // [rsp+118h] [rbp-F8h]
  __int64 v76; // [rsp+120h] [rbp-F0h] BYREF
  __int64 v77; // [rsp+128h] [rbp-E8h] BYREF
  __int64 v78; // [rsp+130h] [rbp-E0h]
  __int64 v79; // [rsp+138h] [rbp-D8h] BYREF
  unsigned int v80; // [rsp+140h] [rbp-D0h]
  char v81; // [rsp+178h] [rbp-98h] BYREF
  int v82; // [rsp+17Ch] [rbp-94h]
  __int64 v83; // [rsp+180h] [rbp-90h]
  __int64 v84; // [rsp+188h] [rbp-88h]
  int v85; // [rsp+190h] [rbp-80h]
  unsigned __int64 v86; // [rsp+198h] [rbp-78h]
  unsigned int v87; // [rsp+1A0h] [rbp-70h]
  __int64 v88; // [rsp+1A8h] [rbp-68h]
  __int64 v89; // [rsp+1B0h] [rbp-60h]
  int v90; // [rsp+1B8h] [rbp-58h]
  unsigned __int64 v91; // [rsp+1C0h] [rbp-50h]
  unsigned int v92; // [rsp+1C8h] [rbp-48h]
  int v93; // [rsp+1D0h] [rbp-40h]
  __int64 v94; // [rsp+1D8h] [rbp-38h]

  v2 = a2 - a1;
  if ( v2 <= 192 )
    return;
  v46 = 0xAAAAAAAAAAAAAAABLL * (v2 >> 6);
  v3 = (v46 - 2) / 2;
  for ( i = a1 + ((v3 + ((v46 - 2 + ((unsigned __int64)(v46 - 2) >> 63)) & 0xFFFFFFFFFFFFFFFELL)) << 6) + 24; ; i -= 192LL )
  {
    v5 = *(_QWORD *)(i - 24);
    v6 = &v60;
    v7 = i - 24;
    v58 = 0;
    v59 = 1;
    v57 = v5;
    v8 = (__int64 *)(i - 16);
    do
      *v6++ = -4096;
    while ( v6 != (__int64 *)v62 );
    v54 = *(_DWORD *)(v7 + 16) >> 1;
    v9 = HIDWORD(v59);
    v10 = v59 & 0xFFFFFFFE | *(_DWORD *)(v7 + 16) & 1;
    LODWORD(v59) = *(_DWORD *)(v7 + 16) & 0xFFFFFFFE | v59 & 1;
    v11 = v59 & 1;
    *(_DWORD *)(v7 + 16) = v10;
    v12 = *(_DWORD *)(i - 4);
    *(_DWORD *)(i - 4) = v9;
    HIDWORD(v59) = v12;
    if ( v11 )
    {
      v13 = &v58;
      if ( (*(_BYTE *)(v7 + 16) & 1) != 0 )
      {
        v39 = &v60;
        v40 = (__int64 *)i;
        do
        {
          v41 = *v39;
          *v39++ = *v40;
          *v40++ = v41;
        }
        while ( v62 != (_BYTE *)v39 );
        goto LABEL_10;
      }
    }
    else
    {
      v13 = (__int64 *)(i - 16);
      v8 = &v58;
      if ( (*(_BYTE *)(v7 + 16) & 1) == 0 )
      {
        v42 = v61;
        v43 = *(_QWORD *)i;
        *(_QWORD *)i = v60;
        v44 = *(_DWORD *)(i + 8);
        v60 = v43;
        v61 = v44;
        *(_DWORD *)(i + 8) = v42;
        goto LABEL_11;
      }
    }
    *((_BYTE *)v8 + 8) |= 1u;
    v14 = v8[2];
    v15 = 2;
    v16 = *((_DWORD *)v8 + 6);
    do
    {
      v8[v15] = v13[v15];
      ++v15;
    }
    while ( v15 != 10 );
    *((_BYTE *)v13 + 8) &= ~1u;
    v13[2] = v14;
    *((_DWORD *)v13 + 6) = v16;
LABEL_10:
    v12 = HIDWORD(v59);
    v54 = (unsigned int)v59 >> 1;
LABEL_11:
    v17 = *(_QWORD *)(i + 112);
    v18 = *(_QWORD *)(i + 120);
    v19 = *(_DWORD *)(i + 128);
    v52 = *(_BYTE *)(i + 64);
    v20 = *(_DWORD *)(i + 144);
    v62[0] = v52;
    v21 = *(_DWORD *)(i + 68);
    v22 = *(_QWORD *)(i + 136);
    v69 = v17;
    v53 = v21;
    v63 = v21;
    v23 = *(_QWORD *)(i + 72);
    v70 = v18;
    v56 = v23;
    v64 = v23;
    v24 = *(_QWORD *)(i + 80);
    v71 = v19;
    v51 = v24;
    v65 = v24;
    LODWORD(v24) = *(_DWORD *)(i + 88);
    v73 = v20;
    v50 = v24;
    v66 = v24;
    LODWORD(v24) = *(_DWORD *)(i + 104);
    v72 = v22;
    v49 = v24;
    v68 = v24;
    v25 = *(_QWORD *)(i + 96);
    *(_DWORD *)(i + 104) = 0;
    v48 = v25;
    v67 = v25;
    *(_DWORD *)(i + 144) = 0;
    v26 = *(_DWORD *)(i + 152);
    v27 = *(_QWORD *)(i + 160);
    v77 = 0;
    v74 = v26;
    v75 = v27;
    v78 = 1;
    v76 = v57;
    v28 = &v79;
    do
      *v28++ = -4096;
    while ( v28 != (__int64 *)&v81 );
    LODWORD(v59) = v78 & 0xFFFFFFFE | v59 & 1;
    v29 = HIDWORD(v78);
    HIDWORD(v78) = v12;
    LODWORD(v78) = (2 * v54) | v78 & 1;
    HIDWORD(v59) = v29;
    if ( (v78 & 1) != 0 )
    {
      v31 = &v58;
      v30 = &v77;
      if ( (v59 & 1) != 0 )
      {
        v36 = &v79;
        v37 = &v60;
        do
        {
          v38 = *v36;
          *v36++ = *v37++;
          *(v37 - 1) = v38;
        }
        while ( v62 != (_BYTE *)v37 );
        goto LABEL_18;
      }
    }
    else
    {
      v30 = &v58;
      v31 = &v77;
      v32 = v59 & 1;
      if ( (v59 & 1) == 0 )
      {
        v55 = v79;
        v79 = v60;
        v60 = v55;
        v45 = v61;
        v61 = v80;
        v80 = v45;
        goto LABEL_19;
      }
    }
    *((_BYTE *)v31 + 8) |= 1u;
    v33 = v31[2];
    v34 = 2;
    v35 = *((_DWORD *)v31 + 6);
    do
    {
      v31[v34] = v30[v34];
      ++v34;
    }
    while ( v34 != 10 );
    *((_BYTE *)v30 + 8) &= ~1u;
    v30[2] = v33;
    *((_DWORD *)v30 + 6) = v35;
LABEL_18:
    v17 = v69;
    v18 = v70;
    v52 = v62[0];
    v32 = v59 & 1;
    v19 = v71;
    v20 = v73;
    v53 = v63;
    v22 = v72;
    v26 = v74;
    v56 = v64;
    v27 = v75;
    v51 = v65;
    v50 = v66;
    v49 = v68;
    v48 = v67;
LABEL_19:
    v91 = v22;
    v94 = v27;
    v81 = v52;
    v93 = v26;
    v82 = v53;
    v88 = v17;
    v83 = v56;
    v89 = v18;
    v84 = v51;
    v90 = v19;
    v85 = v50;
    v92 = v20;
    v87 = v49;
    v86 = v48;
    sub_28B6270(a1, v3, v46, &v76);
    if ( v92 > 0x40 && v91 )
      j_j___libc_free_0_0(v91);
    if ( v87 > 0x40 && v86 )
      j_j___libc_free_0_0(v86);
    if ( (v78 & 1) == 0 )
      sub_C7D6A0(v79, 8LL * v80, 8);
    if ( !v3 )
      break;
    --v3;
    if ( !v32 )
      sub_C7D6A0(v60, 8LL * v61, 8);
  }
  if ( !v32 )
    sub_C7D6A0(v60, 8LL * v61, 8);
}
