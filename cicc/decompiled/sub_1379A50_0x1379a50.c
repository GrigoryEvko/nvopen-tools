// Function: sub_1379A50
// Address: 0x1379a50
//
__int64 __fastcall sub_1379A50(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r13
  int v7; // r15d
  unsigned int v8; // r12d
  _BYTE *v10; // r15
  unsigned int v11; // eax
  unsigned int v12; // r9d
  __int64 v13; // rdx
  unsigned __int64 v14; // r15
  int v15; // r14d
  _BYTE *v16; // r10
  __int64 v17; // rbx
  unsigned int v18; // r13d
  __int64 v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  _BOOL4 v24; // eax
  unsigned int v25; // r11d
  __int64 v26; // rax
  __int64 v27; // rax
  _BYTE *v28; // rax
  unsigned __int64 v29; // r10
  unsigned __int64 v30; // r13
  unsigned int v31; // eax
  __int64 v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rsi
  __int64 v36; // rdx
  unsigned int v37; // eax
  __int64 v38; // rdx
  __int64 v39; // rax
  unsigned int v40; // eax
  __int64 v41; // r12
  __int64 v42; // rbx
  __int64 v43; // rax
  unsigned int *v44; // rcx
  unsigned int v45; // edi
  unsigned int *v46; // r9
  unsigned int v47; // eax
  unsigned int *v48; // rsi
  unsigned int v49; // edx
  __int64 v50; // r10
  unsigned __int64 v51; // r8
  unsigned int *v52; // rcx
  unsigned int *v53; // r8
  unsigned int v54; // eax
  unsigned int *v55; // rdi
  unsigned int v56; // edx
  __int64 v57; // r13
  unsigned int v58; // eax
  __int64 v59; // r12
  int v60; // edx
  int v61; // ecx
  _BYTE *v62; // [rsp+0h] [rbp-100h]
  int v63; // [rsp+0h] [rbp-100h]
  __int64 v64; // [rsp+8h] [rbp-F8h]
  _BYTE *v65; // [rsp+8h] [rbp-F8h]
  __int64 v66; // [rsp+18h] [rbp-E8h]
  unsigned int v67; // [rsp+20h] [rbp-E0h]
  int v68; // [rsp+4Ch] [rbp-B4h] BYREF
  _BYTE *v69; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v70; // [rsp+58h] [rbp-A8h]
  _BYTE v71[16]; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int *v72; // [rsp+70h] [rbp-90h] BYREF
  __int64 v73; // [rsp+78h] [rbp-88h]
  _BYTE v74[16]; // [rsp+80h] [rbp-80h] BYREF
  unsigned int *v75; // [rsp+90h] [rbp-70h] BYREF
  __int64 v76; // [rsp+98h] [rbp-68h]
  _BYTE v77[16]; // [rsp+A0h] [rbp-60h] BYREF
  _BYTE *v78; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v79; // [rsp+B8h] [rbp-48h]
  _BYTE v80[64]; // [rsp+C0h] [rbp-40h] BYREF

  v2 = a1;
  v3 = a2;
  v4 = sub_157EBA0(a2);
  if ( (unsigned __int8)(*(_BYTE *)(v4 + 16) - 26) > 2u || !*(_QWORD *)(v4 + 48) && *(__int16 *)(v4 + 18) >= 0 )
    return 0;
  v5 = sub_1625790(v4, 2);
  v6 = v5;
  if ( !v5 )
    return 0;
  v7 = *(_DWORD *)(v5 + 8);
  if ( (unsigned int)sub_15F4D60(v4) + 1 != v7 )
    return 0;
  v10 = v71;
  v72 = (unsigned int *)v74;
  v69 = v71;
  v70 = 0x200000000LL;
  v73 = 0x200000000LL;
  v75 = (unsigned int *)v77;
  v76 = 0x200000000LL;
  v11 = sub_15F4D60(v4);
  if ( v11 > 2 )
    sub_16CD150(&v69, v71, v11, 4);
  v12 = *(_DWORD *)(v6 + 8);
  v13 = v12;
  if ( v12 != 1 )
  {
    v14 = 0;
    v15 = *(_DWORD *)(v6 + 8);
    v16 = v71;
    v17 = v6;
    v18 = 1;
    while ( 1 )
    {
      v19 = *(_QWORD *)(v17 + 8 * (v18 - v13));
      if ( *(_BYTE *)v19 != 1 || (v20 = *(_QWORD *)(v19 + 136), *(_BYTE *)(v20 + 16) != 13) )
      {
        v10 = v16;
        v8 = 0;
        goto LABEL_26;
      }
      v21 = *(_QWORD **)(v20 + 24);
      if ( *(_DWORD *)(v20 + 32) > 0x40u )
        v21 = (_QWORD *)*v21;
      v22 = (unsigned int)v70;
      if ( (unsigned int)v70 >= HIDWORD(v70) )
      {
        v63 = (int)v21;
        v65 = v16;
        sub_16CD150(&v69, v16, 0, 4);
        v22 = (unsigned int)v70;
        LODWORD(v21) = v63;
        v16 = v65;
      }
      v62 = v16;
      *(_DWORD *)&v69[4 * v22] = (_DWORD)v21;
      LODWORD(v70) = v70 + 1;
      v14 += *(unsigned int *)&v69[4 * (unsigned int)v70 - 4];
      v23 = sub_15F4DF0(v4, v18 - 1);
      v24 = sub_1377F70(a1 + 72, v23);
      v25 = v18 - 1;
      v16 = v62;
      if ( v24 )
      {
        v27 = (unsigned int)v73;
        if ( (unsigned int)v73 >= HIDWORD(v73) )
        {
          sub_16CD150(&v72, v74, 0, 4);
          v27 = (unsigned int)v73;
          v16 = v62;
          v25 = v18 - 1;
        }
        v72[v27] = v25;
        LODWORD(v73) = v73 + 1;
      }
      else
      {
        v26 = (unsigned int)v76;
        if ( (unsigned int)v76 >= HIDWORD(v76) )
        {
          sub_16CD150(&v75, v77, 0, 4);
          v26 = (unsigned int)v76;
          v16 = v62;
          v25 = v18 - 1;
        }
        v75[v26] = v25;
        LODWORD(v76) = v76 + 1;
      }
      if ( ++v18 == v15 )
        break;
      v13 = *(unsigned int *)(v17 + 8);
    }
    v28 = v16;
    v29 = v14;
    v2 = a1;
    v3 = a2;
    v10 = v28;
    if ( v29 <= 0xFFFFFFFF )
    {
LABEL_39:
      if ( v29 && (_DWORD)v76 )
        goto LABEL_45;
      goto LABEL_41;
    }
    v30 = v29 / 0xFFFFFFFF + 1;
    v31 = sub_15F4D60(v4);
    v32 = v31;
    if ( v31 )
    {
      v33 = (unsigned __int64)v69;
      v34 = 4 * v32;
      v35 = 0;
      v29 = 0;
      do
      {
        *(_DWORD *)(v33 + v35) = *(unsigned int *)(v33 + v35) / v30;
        v33 = (unsigned __int64)v69;
        v36 = *(unsigned int *)&v69[v35];
        v35 += 4;
        v29 += v36;
      }
      while ( v34 != v35 );
      goto LABEL_39;
    }
  }
LABEL_41:
  v37 = sub_15F4D60(v4);
  if ( v37 )
  {
    v38 = 4LL * v37;
    v39 = 0;
    do
    {
      *(_DWORD *)&v69[v39] = 1;
      v39 += 4;
    }
    while ( v38 != v39 );
  }
  LODWORD(v29) = sub_15F4D60(v4);
LABEL_45:
  v67 = v29;
  v78 = v80;
  v79 = 0x200000000LL;
  v40 = sub_15F4D60(v4);
  if ( v40 )
  {
    v66 = v4;
    v41 = 4LL * v40;
    v64 = v3;
    v42 = 0;
    do
    {
      sub_16AF710(&v68, *(unsigned int *)&v69[v42], v67);
      v43 = (unsigned int)v79;
      if ( (unsigned int)v79 >= HIDWORD(v79) )
      {
        sub_16CD150(&v78, v80, 0, 4);
        v43 = (unsigned int)v79;
      }
      v42 += 4;
      *(_DWORD *)&v78[4 * v43] = v68;
      LODWORD(v79) = v79 + 1;
    }
    while ( v41 != v42 );
    v4 = v66;
    v3 = v64;
  }
  if ( (_DWORD)v73 && (_DWORD)v76 )
  {
    v44 = v72;
    v45 = dword_4F98720;
    v46 = &v72[(unsigned int)v73];
    v47 = 0;
    do
    {
      v48 = (unsigned int *)&v78[4 * *v44];
      v49 = *v48;
      if ( v45 < *v48 )
      {
        *v48 = v45;
        v50 = v49 - v45;
        v51 = v50 + v47;
        v47 += v50;
        if ( v51 > 0x80000000 )
          v47 = 0x80000000;
      }
      ++v44;
    }
    while ( v46 != v44 );
    if ( v47 )
    {
      v52 = v75;
      v53 = &v75[(unsigned int)v76];
      v54 = v47 / (unsigned int)v76;
      if ( v53 != v75 )
      {
        do
        {
          v55 = (unsigned int *)&v78[4 * *v52];
          v56 = v54 + *v55;
          if ( v54 + (unsigned __int64)*v55 > 0x80000000 )
            v56 = 0x80000000;
          ++v52;
          *v55 = v56;
        }
        while ( v53 != v52 );
      }
    }
  }
  v57 = 0;
  v58 = sub_15F4D60(v4);
  v59 = v58;
  if ( v58 )
  {
    do
    {
      v60 = v57;
      v61 = *(_DWORD *)&v78[4 * v57++];
      sub_1379150(v2, v3, v60, v61);
    }
    while ( v57 != v59 );
  }
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
  v8 = 1;
LABEL_26:
  if ( v75 != (unsigned int *)v77 )
    _libc_free((unsigned __int64)v75);
  if ( v72 != (unsigned int *)v74 )
    _libc_free((unsigned __int64)v72);
  if ( v69 != v10 )
    _libc_free((unsigned __int64)v69);
  return v8;
}
