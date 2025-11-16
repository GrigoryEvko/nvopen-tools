// Function: sub_1430F70
// Address: 0x1430f70
//
__int64 __fastcall sub_1430F70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  bool v6; // r14
  unsigned __int8 v7; // r15
  char v8; // bl
  char v9; // bl
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // r9
  __int64 v14; // rsi
  __int64 v15; // rax
  char v16; // bl
  char v17; // r15
  __int64 v19; // rsi
  __int64 v20; // rbx
  __int64 v21; // rbx
  __int64 v22; // rax
  unsigned int v23; // ecx
  _QWORD *v24; // rdi
  unsigned int v25; // eax
  int v26; // eax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // r9
  _QWORD *v30; // rax
  _QWORD *i; // rdx
  _QWORD *v32; // rax
  __int64 v34; // [rsp+10h] [rbp-1D0h]
  __int64 v35; // [rsp+18h] [rbp-1C8h]
  __int64 v36; // [rsp+18h] [rbp-1C8h]
  int v37; // [rsp+18h] [rbp-1C8h]
  __int64 v38; // [rsp+20h] [rbp-1C0h]
  __int64 v39; // [rsp+20h] [rbp-1C0h]
  __int64 v40; // [rsp+20h] [rbp-1C0h]
  char v41; // [rsp+2Fh] [rbp-1B1h]
  __int64 v42; // [rsp+30h] [rbp-1B0h] BYREF
  __int64 v43[2]; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 v44; // [rsp+50h] [rbp-190h] BYREF
  __int64 v45; // [rsp+60h] [rbp-180h] BYREF
  _QWORD *v46; // [rsp+68h] [rbp-178h]
  __int64 v47; // [rsp+70h] [rbp-170h]
  __int64 v48; // [rsp+78h] [rbp-168h]
  __int64 v49; // [rsp+80h] [rbp-160h]
  __int64 v50; // [rsp+88h] [rbp-158h]
  __int64 v51; // [rsp+90h] [rbp-150h]
  __int64 v52; // [rsp+A0h] [rbp-140h] BYREF
  _BYTE *v53; // [rsp+A8h] [rbp-138h]
  _BYTE *v54; // [rsp+B0h] [rbp-130h]
  __int64 v55; // [rsp+B8h] [rbp-128h]
  int v56; // [rsp+C0h] [rbp-120h]
  _BYTE v57[72]; // [rsp+C8h] [rbp-118h] BYREF
  _QWORD v58[26]; // [rsp+110h] [rbp-D0h] BYREF

  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = v57;
  v54 = v57;
  v55 = 8;
  v56 = 0;
  v41 = sub_1430880(a1, a2, (__int64)&v45, (__int64)&v52);
  sub_15E64D0(a2);
  if ( v5 )
  {
    v7 = *(_BYTE *)(a2 + 32) & 0xF;
    v6 = (unsigned int)v7 - 7 <= 1;
  }
  else
  {
    v6 = 0;
    v7 = *(_BYTE *)(a2 + 32) & 0xF;
  }
  v8 = *(_BYTE *)(a2 + 33);
  ++v45;
  v9 = (v8 & 0x40) != 0;
  if ( !(_DWORD)v47 )
  {
    if ( !HIDWORD(v47) )
      goto LABEL_9;
    v10 = (unsigned int)v48;
    if ( (unsigned int)v48 > 0x40 )
    {
      j___libc_free_0(v46);
      v46 = 0;
      v47 = 0;
      LODWORD(v48) = 0;
      goto LABEL_9;
    }
    goto LABEL_6;
  }
  v23 = 4 * v47;
  v10 = (unsigned int)v48;
  if ( (unsigned int)(4 * v47) < 0x40 )
    v23 = 64;
  if ( (unsigned int)v48 <= v23 )
  {
LABEL_6:
    v11 = v46;
    v12 = &v46[v10];
    if ( v46 != v12 )
    {
      do
        *v11++ = -8;
      while ( v12 != v11 );
    }
    v47 = 0;
    goto LABEL_9;
  }
  v24 = v46;
  if ( (_DWORD)v47 == 1 )
  {
    v37 = 128;
    v29 = 1024;
LABEL_34:
    v40 = v29;
    j___libc_free_0(v46);
    LODWORD(v48) = v37;
    v30 = (_QWORD *)sub_22077B0(v40);
    v47 = 0;
    v46 = v30;
    for ( i = &v30[(unsigned int)v48]; i != v30; ++v30 )
    {
      if ( v30 )
        *v30 = -8;
    }
    goto LABEL_9;
  }
  _BitScanReverse(&v25, v47 - 1);
  v26 = 1 << (33 - (v25 ^ 0x1F));
  if ( v26 < 64 )
    v26 = 64;
  if ( (_DWORD)v48 != v26 )
  {
    v27 = (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
        | (4 * v26 / 3u + 1)
        | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)
        | (((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
          | (4 * v26 / 3u + 1)
          | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4);
    v28 = (v27 >> 8) | v27;
    v37 = (v28 | (v28 >> 16)) + 1;
    v29 = 8 * ((v28 | (v28 >> 16)) + 1);
    goto LABEL_34;
  }
  v47 = 0;
  v32 = &v46[(unsigned int)v48];
  do
  {
    if ( v24 )
      *v24 = -8;
    ++v24;
  }
  while ( v32 != v24 );
LABEL_9:
  v13 = v49;
  v14 = v51;
  v51 = 0;
  v49 = 0;
  v34 = v13;
  v35 = v50;
  v50 = 0;
  v15 = sub_22077B0(64);
  if ( v15 )
  {
    *(_DWORD *)(v15 + 8) = 2;
    *(_QWORD *)(v15 + 16) = 0;
    *(_QWORD *)(v15 + 48) = v35;
    v16 = (16 * v6) | v7 | (v9 << 6);
    v17 = *(_BYTE *)(v15 + 12);
    *(_QWORD *)(v15 + 24) = 0;
    *(_QWORD *)(v15 + 32) = 0;
    *(_QWORD *)(v15 + 40) = v34;
    *(_QWORD *)(v15 + 56) = v14;
    *(_BYTE *)(v15 + 12) = v17 & 0x80 | v16;
    *(_QWORD *)v15 = &unk_49EB4D8;
  }
  else
  {
    v19 = v14 - v34;
    if ( v34 )
    {
      j_j___libc_free_0(v34, v19);
      v15 = 0;
      if ( !v6 )
        goto LABEL_12;
      goto LABEL_23;
    }
  }
  if ( !v6 )
    goto LABEL_12;
LABEL_23:
  v36 = v15;
  sub_15E4EB0(v43);
  v20 = v43[1];
  v38 = v43[0];
  sub_16C1840(v58);
  sub_16C1A90(v58, v38, v20);
  sub_16C1AA0(v58, &v42);
  v21 = v42;
  v22 = v36;
  if ( (__int64 *)v43[0] != &v44 )
  {
    j_j___libc_free_0(v43[0], v44 + 1);
    v22 = v36;
  }
  v39 = v22;
  v43[0] = v21;
  sub_142F900((__int64)v58, a3, v43);
  v15 = v39;
LABEL_12:
  if ( v41 )
    *(_BYTE *)(v15 + 12) |= 0x10u;
  v58[0] = v15;
  sub_142ED30(a1, a2, v58);
  if ( v58[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v58[0] + 8LL))(v58[0]);
  if ( v54 != v53 )
    _libc_free((unsigned __int64)v54);
  if ( v49 )
    j_j___libc_free_0(v49, v51 - v49);
  return j___libc_free_0(v46);
}
