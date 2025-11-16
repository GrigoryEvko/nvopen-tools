// Function: sub_28B7290
// Address: 0x28b7290
//
__int64 __fastcall sub_28B7290(__int64 a1, __int64 a2)
{
  __int64 *v2; // rcx
  __int64 v4; // rax
  __int64 *v5; // rax
  unsigned int v6; // r10d
  unsigned int v7; // eax
  int v8; // edx
  int v9; // esi
  __int64 *v10; // rsi
  __int64 v11; // r9
  __int64 v12; // rax
  int v13; // r8d
  char v14; // r9
  __int64 *v15; // r13
  __int64 v16; // r8
  int v17; // ecx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // r15d
  __int64 v22; // r14
  __int64 v23; // rax
  bool v24; // zf
  int v25; // ecx
  __int64 v26; // r8
  char v27; // r9
  unsigned int v28; // r10d
  __int64 *v29; // rax
  __int64 *v30; // rsi
  _QWORD *v31; // rdx
  int v32; // r11d
  int v33; // edi
  char v34; // r12
  __int64 *v35; // rdx
  __int64 *v36; // rcx
  __int64 v37; // r8
  __int64 v38; // rax
  int v39; // edi
  bool v40; // cc
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdi
  __int64 result; // rax
  __int64 *v44; // rax
  __int64 *v45; // rdx
  __int64 v46; // rsi
  __int64 *v47; // rdx
  __int64 v48; // rsi
  unsigned int v49; // eax
  __int64 v50; // rcx
  unsigned int v51; // edx
  unsigned int v52; // eax
  __int64 v53; // rsi
  unsigned int v54; // edx
  unsigned int v55; // [rsp+4h] [rbp-14Ch]
  int v56; // [rsp+4h] [rbp-14Ch]
  char v57; // [rsp+8h] [rbp-148h]
  __int64 v58; // [rsp+8h] [rbp-148h]
  __int64 v59; // [rsp+10h] [rbp-140h]
  unsigned int v60; // [rsp+10h] [rbp-140h]
  int v61; // [rsp+20h] [rbp-130h]
  char v62; // [rsp+20h] [rbp-130h]
  unsigned int v63; // [rsp+24h] [rbp-12Ch]
  __int64 v64; // [rsp+28h] [rbp-128h]
  __int64 v65; // [rsp+30h] [rbp-120h]
  __int64 v66; // [rsp+38h] [rbp-118h]
  int v67; // [rsp+40h] [rbp-110h]
  int v68; // [rsp+44h] [rbp-10Ch]
  __int64 v69; // [rsp+48h] [rbp-108h]
  __int64 v70; // [rsp+50h] [rbp-100h]
  int v71; // [rsp+58h] [rbp-F8h]
  int v72; // [rsp+5Ch] [rbp-F4h]
  __int64 v73; // [rsp+60h] [rbp-F0h]
  __int64 v74; // [rsp+68h] [rbp-E8h] BYREF
  unsigned __int64 v75; // [rsp+70h] [rbp-E0h]
  __int64 v76; // [rsp+78h] [rbp-D8h] BYREF
  unsigned int v77; // [rsp+80h] [rbp-D0h]
  _BYTE v78[4]; // [rsp+B8h] [rbp-98h] BYREF
  int v79; // [rsp+BCh] [rbp-94h]
  __int64 v80; // [rsp+C0h] [rbp-90h]
  __int64 v81; // [rsp+C8h] [rbp-88h]
  int v82; // [rsp+D0h] [rbp-80h]
  __int64 v83; // [rsp+D8h] [rbp-78h]
  int v84; // [rsp+E0h] [rbp-70h]
  __int64 v85; // [rsp+E8h] [rbp-68h]
  __int64 v86; // [rsp+F0h] [rbp-60h]
  int v87; // [rsp+F8h] [rbp-58h]
  __int64 v88; // [rsp+100h] [rbp-50h]
  unsigned int v89; // [rsp+108h] [rbp-48h]
  int v90; // [rsp+110h] [rbp-40h]
  __int64 v91; // [rsp+118h] [rbp-38h]

  v2 = (__int64 *)(a1 + 8);
  v4 = *(_QWORD *)a1;
  v74 = 0;
  v75 = 1;
  v73 = v4;
  v5 = &v76;
  do
    *v5++ = -4096;
  while ( v5 != (__int64 *)v78 );
  v6 = *(_DWORD *)(a1 + 20);
  v72 = *(_DWORD *)(a1 + 16) >> 1;
  v7 = v75 & 1 | *(_DWORD *)(a1 + 16) & 0xFFFFFFFE;
  v8 = HIDWORD(v75);
  v9 = v75 & 0xFFFFFFFE | *(_DWORD *)(a1 + 16) & 1;
  v75 = __PAIR64__(v6, v7);
  *(_DWORD *)(a1 + 16) = v9;
  *(_DWORD *)(a1 + 20) = v8;
  if ( (v7 & 1) != 0 )
  {
    v10 = &v74;
    if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
    {
      v44 = &v76;
      v45 = (__int64 *)(a1 + 24);
      do
      {
        v46 = *v44;
        *v44++ = *v45;
        *v45++ = v46;
      }
      while ( v44 != (__int64 *)v78 );
      goto LABEL_8;
    }
LABEL_5:
    *((_BYTE *)v2 + 8) |= 1u;
    v11 = v2[2];
    v12 = 2;
    v13 = *((_DWORD *)v2 + 6);
    do
    {
      v2[v12] = v10[v12];
      ++v12;
    }
    while ( v12 != 10 );
    *((_BYTE *)v10 + 8) &= ~1u;
    v10[2] = v11;
    *((_DWORD *)v10 + 6) = v13;
LABEL_8:
    v6 = HIDWORD(v75);
    v72 = (unsigned int)v75 >> 1;
    goto LABEL_9;
  }
  v10 = (__int64 *)(a1 + 8);
  v2 = &v74;
  if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
    goto LABEL_5;
  v49 = v77;
  v50 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(a1 + 24) = v76;
  v51 = *(_DWORD *)(a1 + 32);
  v76 = v50;
  v77 = v51;
  *(_DWORD *)(a1 + 32) = v49;
LABEL_9:
  v14 = *(_BYTE *)(a1 + 88);
  v15 = (__int64 *)(a2 + 8);
  v16 = *(_QWORD *)(a1 + 104);
  v17 = *(_DWORD *)(a1 + 112);
  v55 = v6;
  v71 = *(_DWORD *)(a1 + 92);
  v79 = v71;
  v18 = *(_QWORD *)(a1 + 96);
  v78[0] = v14;
  v70 = v18;
  v80 = v18;
  v57 = v14;
  v68 = *(_DWORD *)(a1 + 128);
  v84 = v68;
  v19 = *(_QWORD *)(a1 + 120);
  v81 = v16;
  v69 = v19;
  v83 = v19;
  v59 = v16;
  v66 = *(_QWORD *)(a1 + 136);
  v85 = v66;
  v20 = *(_QWORD *)(a1 + 144);
  v82 = v17;
  v65 = v20;
  v86 = v20;
  LODWORD(v20) = *(_DWORD *)(a1 + 152);
  v61 = v17;
  *(_DWORD *)(a1 + 128) = 0;
  v67 = v20;
  v87 = v20;
  LODWORD(v20) = *(_DWORD *)(a1 + 168);
  v21 = *(_DWORD *)(a1 + 176);
  v22 = *(_QWORD *)(a1 + 184);
  *(_DWORD *)(a1 + 168) = 0;
  v63 = v20;
  v89 = v20;
  v23 = *(_QWORD *)(a1 + 160);
  v90 = v21;
  v64 = v23;
  v88 = v23;
  v91 = v22;
  sub_28B56D0(a1, a2);
  v24 = (*(_BYTE *)(a2 + 16) & 1) == 0;
  v25 = v61;
  v26 = v59;
  *(_QWORD *)a2 = v73;
  v27 = v57;
  v28 = v55;
  if ( v24 )
  {
    v56 = v61;
    v58 = v59;
    v60 = v28;
    v62 = v27;
    sub_C7D6A0(*(_QWORD *)(a2 + 24), 8LL * *(unsigned int *)(a2 + 32), 8);
    v25 = v56;
    v26 = v58;
    v28 = v60;
    v27 = v62;
  }
  *(_QWORD *)(a2 + 16) = 1;
  v29 = (__int64 *)(a2 + 24);
  v30 = (__int64 *)(a2 + 24);
  v31 = (_QWORD *)(a2 + 24);
  do
  {
    if ( v31 )
      *v31 = -4096;
    ++v31;
  }
  while ( (_QWORD *)(a2 + 88) != v31 );
  v32 = *(_DWORD *)(a2 + 16);
  LODWORD(v75) = v32 & 0xFFFFFFFE | v75 & 1;
  v33 = *(_DWORD *)(a2 + 20);
  *(_DWORD *)(a2 + 20) = v28;
  *(_DWORD *)(a2 + 16) = (2 * v72) | v32 & 1;
  HIDWORD(v75) = v33;
  if ( (v32 & 1) != 0 )
  {
    v36 = &v76;
    v35 = &v74;
    if ( (v75 & 1) != 0 )
    {
      v47 = &v76;
      do
      {
        v48 = *v29;
        *v29++ = *v47++;
        *(v47 - 1) = v48;
      }
      while ( v78 != (_BYTE *)v47 );
      goto LABEL_21;
    }
  }
  else
  {
    v34 = v75 & 1;
    if ( (v75 & 1) == 0 )
    {
      v52 = *(_DWORD *)(a2 + 32);
      v53 = v76;
      v76 = *(_QWORD *)(a2 + 24);
      v54 = v77;
      *(_QWORD *)(a2 + 24) = v53;
      *(_DWORD *)(a2 + 32) = v54;
      v77 = v52;
      goto LABEL_22;
    }
    v30 = &v76;
    v35 = (__int64 *)(a2 + 8);
    v36 = (__int64 *)(a2 + 24);
    v15 = &v74;
  }
  *((_BYTE *)v35 + 8) |= 1u;
  v37 = v35[2];
  v38 = 0;
  v39 = *((_DWORD *)v35 + 6);
  do
  {
    v36[v38] = v30[v38];
    ++v38;
  }
  while ( v38 != 8 );
  *((_BYTE *)v15 + 8) &= ~1u;
  v15[2] = v37;
  *((_DWORD *)v15 + 6) = v39;
LABEL_21:
  v27 = v78[0];
  v21 = v90;
  v71 = v79;
  v34 = v75 & 1;
  v22 = v91;
  v26 = v81;
  v70 = v80;
  v25 = v82;
  v69 = v83;
  v68 = v84;
  v66 = v85;
  v65 = v86;
  v67 = v87;
  v64 = v88;
  v63 = v89;
LABEL_22:
  v40 = *(_DWORD *)(a2 + 128) <= 0x40u;
  *(_BYTE *)(a2 + 88) = v27;
  *(_QWORD *)(a2 + 104) = v26;
  *(_DWORD *)(a2 + 92) = v71;
  *(_DWORD *)(a2 + 112) = v25;
  *(_QWORD *)(a2 + 96) = v70;
  if ( !v40 )
  {
    v41 = *(_QWORD *)(a2 + 120);
    if ( v41 )
      j_j___libc_free_0_0(v41);
  }
  v40 = *(_DWORD *)(a2 + 168) <= 0x40u;
  *(_QWORD *)(a2 + 120) = v69;
  *(_DWORD *)(a2 + 128) = v68;
  *(_QWORD *)(a2 + 136) = v66;
  *(_QWORD *)(a2 + 144) = v65;
  *(_DWORD *)(a2 + 152) = v67;
  if ( !v40 )
  {
    v42 = *(_QWORD *)(a2 + 160);
    if ( v42 )
      j_j___libc_free_0_0(v42);
  }
  *(_DWORD *)(a2 + 176) = v21;
  *(_QWORD *)(a2 + 184) = v22;
  *(_QWORD *)(a2 + 160) = v64;
  result = v63;
  *(_DWORD *)(a2 + 168) = v63;
  if ( !v34 )
    return sub_C7D6A0(v76, 8LL * v77, 8);
  return result;
}
