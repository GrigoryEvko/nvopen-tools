// Function: sub_30D34F0
// Address: 0x30d34f0
//
__int64 __fastcall sub_30D34F0(__int64 a1)
{
  __int64 v2; // rax
  unsigned int v3; // esi
  int v4; // edx
  __int64 v6; // rax
  __int64 v7; // rsi
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // r12
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  unsigned int v12; // ecx
  __int64 v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  unsigned __int64 v16; // r13
  __int64 v17; // r15
  __int64 *v18; // r12
  __int64 *v19; // r14
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rdx
  char v25; // al
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  __int64 *v28; // r12
  __int64 *v29; // r13
  __int64 v30; // rsi
  __int64 v31; // rdi
  __int64 *v32; // rax
  __int64 *v33; // r12
  __int64 *v34; // r14
  __int64 v35; // rdi
  unsigned int v36; // ecx
  __int64 v37; // rsi
  __int64 *v38; // r12
  __int64 *v39; // r13
  __int64 v40; // rsi
  __int64 v41; // rdi
  _BYTE *v42; // r13
  _BYTE *v43; // r12
  unsigned __int64 v44; // r14
  unsigned __int64 v45; // rdi
  _QWORD *v46; // rdi
  __int64 v47; // r8
  unsigned int v48; // eax
  int v49; // r12d
  unsigned int v50; // eax
  _QWORD *v51; // rax
  _QWORD *i; // rdx
  _QWORD *v53; // r8
  unsigned __int64 v54; // [rsp+18h] [rbp-158h]
  unsigned __int64 v55[2]; // [rsp+20h] [rbp-150h] BYREF
  char v56; // [rsp+30h] [rbp-140h] BYREF
  _BYTE *v57; // [rsp+38h] [rbp-138h]
  __int64 v58; // [rsp+40h] [rbp-130h]
  _BYTE v59[56]; // [rsp+48h] [rbp-128h] BYREF
  __int64 v60; // [rsp+80h] [rbp-F0h]
  __int64 v61; // [rsp+88h] [rbp-E8h]
  char v62; // [rsp+90h] [rbp-E0h]
  __int64 v63; // [rsp+94h] [rbp-DCh]
  __int64 v64; // [rsp+A0h] [rbp-D0h] BYREF
  _QWORD *v65; // [rsp+A8h] [rbp-C8h]
  __int64 v66; // [rsp+B0h] [rbp-C0h]
  unsigned int v67; // [rsp+B8h] [rbp-B8h]
  unsigned __int64 v68; // [rsp+C0h] [rbp-B0h]
  unsigned __int64 v69; // [rsp+C8h] [rbp-A8h]
  __int64 v70; // [rsp+D8h] [rbp-98h]
  __int64 j; // [rsp+E0h] [rbp-90h]
  __int64 *v72; // [rsp+E8h] [rbp-88h]
  unsigned int v73; // [rsp+F0h] [rbp-80h]
  char v74; // [rsp+F8h] [rbp-78h] BYREF
  __int64 *v75; // [rsp+118h] [rbp-58h]
  unsigned int v76; // [rsp+120h] [rbp-50h]
  __int64 v77; // [rsp+128h] [rbp-48h] BYREF

  v2 = sub_B43CB0(*(_QWORD *)(a1 + 96));
  if ( !(unsigned __int8)sub_B2D610(v2, 18) )
    goto LABEL_2;
  v6 = *(_QWORD *)(a1 + 72);
  v63 = 0;
  v55[0] = (unsigned __int64)&v56;
  v55[1] = 0x100000000LL;
  v60 = 0;
  v62 = 0;
  v61 = v6;
  LODWORD(v6) = *(_DWORD *)(v6 + 92);
  v57 = v59;
  v58 = 0x600000000LL;
  HIDWORD(v63) = v6;
  sub_B1F440((__int64)v55);
  v7 = (__int64)v55;
  sub_D51D90((__int64)&v64, (__int64)v55);
  v8 = v69;
  v9 = v68;
  if ( v68 != v69 )
  {
    while ( 1 )
    {
      v7 = **(_QWORD **)(*(_QWORD *)v9 + 32LL);
      if ( *(_BYTE *)(a1 + 292) )
      {
        v10 = *(_QWORD **)(a1 + 272);
        v11 = &v10[*(unsigned int *)(a1 + 284)];
        if ( v10 == v11 )
          goto LABEL_71;
        while ( v7 != *v10 )
        {
          if ( v11 == ++v10 )
            goto LABEL_71;
        }
LABEL_13:
        v9 += 8LL;
        if ( v8 == v9 )
          break;
      }
      else
      {
        if ( sub_C8CA60(a1 + 264, v7) )
          goto LABEL_13;
LABEL_71:
        v9 += 8LL;
        *(_DWORD *)(a1 + 700) += 25;
        if ( v8 == v9 )
          break;
      }
    }
  }
  ++v64;
  if ( !(_DWORD)v66 )
  {
    if ( !HIDWORD(v66) )
      goto LABEL_21;
    v13 = v67;
    if ( v67 <= 0x40 )
    {
LABEL_18:
      v14 = v65;
      v15 = &v65[2 * v13];
      if ( v65 != v15 )
      {
        do
        {
          *v14 = -4096;
          v14 += 2;
        }
        while ( v15 != v14 );
      }
      goto LABEL_20;
    }
    v7 = 16LL * v67;
    sub_C7D6A0((__int64)v65, v7, 8);
    v67 = 0;
    goto LABEL_76;
  }
  v12 = 4 * v66;
  v7 = 64;
  v13 = v67;
  if ( (unsigned int)(4 * v66) < 0x40 )
    v12 = 64;
  if ( v67 <= v12 )
    goto LABEL_18;
  v46 = v65;
  v47 = 2LL * v67;
  if ( (_DWORD)v66 == 1 )
  {
    v49 = 64;
    goto LABEL_84;
  }
  _BitScanReverse(&v48, v66 - 1);
  v49 = 1 << (33 - (v48 ^ 0x1F));
  if ( v49 < 64 )
    v49 = 64;
  if ( v49 != v67 )
  {
LABEL_84:
    v7 = 16LL * v67;
    sub_C7D6A0((__int64)v65, v47 * 8, 8);
    v50 = sub_30D1A00(v49);
    v67 = v50;
    if ( v50 )
    {
      v7 = 8;
      v51 = (_QWORD *)sub_C7D670(16LL * v50, 8);
      v66 = 0;
      v65 = v51;
      for ( i = &v51[2 * v67]; i != v51; v51 += 2 )
      {
        if ( v51 )
          *v51 = -4096;
      }
      goto LABEL_21;
    }
LABEL_76:
    v65 = 0;
LABEL_20:
    v66 = 0;
    goto LABEL_21;
  }
  v66 = 0;
  v53 = &v65[v47];
  do
  {
    if ( v46 )
      *v46 = -4096;
    v46 += 2;
  }
  while ( v53 != v46 );
LABEL_21:
  v16 = v68;
  v54 = v69;
  if ( v68 != v69 )
  {
    do
    {
      v17 = *(_QWORD *)v16;
      v18 = *(__int64 **)(*(_QWORD *)v16 + 16LL);
      if ( *(__int64 **)(*(_QWORD *)v16 + 8LL) == v18 )
      {
        *(_BYTE *)(v17 + 152) = 1;
      }
      else
      {
        v19 = *(__int64 **)(*(_QWORD *)v16 + 8LL);
        do
        {
          v20 = *v19++;
          sub_D47BB0(v20, v7);
        }
        while ( v18 != v19 );
        *(_BYTE *)(v17 + 152) = 1;
        v21 = *(_QWORD *)(v17 + 8);
        if ( *(_QWORD *)(v17 + 16) != v21 )
          *(_QWORD *)(v17 + 16) = v21;
      }
      v22 = *(_QWORD *)(v17 + 32);
      if ( v22 != *(_QWORD *)(v17 + 40) )
        *(_QWORD *)(v17 + 40) = v22;
      ++*(_QWORD *)(v17 + 56);
      if ( *(_BYTE *)(v17 + 84) )
      {
        *(_QWORD *)v17 = 0;
      }
      else
      {
        v23 = 4 * (*(_DWORD *)(v17 + 76) - *(_DWORD *)(v17 + 80));
        v24 = *(unsigned int *)(v17 + 72);
        if ( v23 < 0x20 )
          v23 = 32;
        if ( (unsigned int)v24 > v23 )
        {
          sub_C8C990(v17 + 56, v7);
        }
        else
        {
          v7 = 0xFFFFFFFFLL;
          memset(*(void **)(v17 + 64), -1, 8 * v24);
        }
        v25 = *(_BYTE *)(v17 + 84);
        *(_QWORD *)v17 = 0;
        if ( !v25 )
          _libc_free(*(_QWORD *)(v17 + 64));
      }
      v26 = *(_QWORD *)(v17 + 32);
      if ( v26 )
      {
        v7 = *(_QWORD *)(v17 + 48) - v26;
        j_j___libc_free_0(v26);
      }
      v27 = *(_QWORD *)(v17 + 8);
      if ( v27 )
      {
        v7 = *(_QWORD *)(v17 + 24) - v27;
        j_j___libc_free_0(v27);
      }
      v16 += 8LL;
    }
    while ( v54 != v16 );
    if ( v68 != v69 )
      v69 = v68;
  }
  v28 = v75;
  v29 = &v75[2 * v76];
  if ( v75 != v29 )
  {
    do
    {
      v30 = v28[1];
      v31 = *v28;
      v28 += 2;
      sub_C7D6A0(v31, v30, 16);
    }
    while ( v29 != v28 );
  }
  v76 = 0;
  if ( v73 )
  {
    v32 = v72;
    v77 = 0;
    v33 = &v72[v73];
    v34 = v72 + 1;
    v70 = *v72;
    for ( j = v70 + 4096; v33 != v34; v32 = v72 )
    {
      v35 = *v34;
      v36 = (unsigned int)(v34 - v32) >> 7;
      v37 = 4096LL << v36;
      if ( v36 >= 0x1E )
        v37 = 0x40000000000LL;
      ++v34;
      sub_C7D6A0(v35, v37, 16);
    }
    v73 = 1;
    sub_C7D6A0(*v32, 4096, 16);
    v38 = v75;
    v39 = &v75[2 * v76];
    if ( v75 == v39 )
      goto LABEL_53;
    do
    {
      v40 = v38[1];
      v41 = *v38;
      v38 += 2;
      sub_C7D6A0(v41, v40, 16);
    }
    while ( v39 != v38 );
  }
  v39 = v75;
LABEL_53:
  if ( v39 != &v77 )
    _libc_free((unsigned __int64)v39);
  if ( v72 != (__int64 *)&v74 )
    _libc_free((unsigned __int64)v72);
  if ( v68 )
    j_j___libc_free_0(v68);
  sub_C7D6A0((__int64)v65, 16LL * v67, 8);
  v42 = v57;
  v43 = &v57[8 * (unsigned int)v58];
  if ( v57 != v43 )
  {
    do
    {
      v44 = *((_QWORD *)v43 - 1);
      v43 -= 8;
      if ( v44 )
      {
        v45 = *(_QWORD *)(v44 + 24);
        if ( v45 != v44 + 40 )
          _libc_free(v45);
        j_j___libc_free_0(v44);
      }
    }
    while ( v42 != v43 );
    v43 = v57;
  }
  if ( v43 != v59 )
    _libc_free((unsigned __int64)v43);
  if ( (char *)v55[0] != &v56 )
    _libc_free(v55[0]);
LABEL_2:
  *(_DWORD *)(a1 + 704) = *(_DWORD *)(a1 + 284) - *(_DWORD *)(a1 + 288);
  v3 = *(_DWORD *)(a1 + 132);
  v4 = *(_DWORD *)(a1 + 760);
  *(_DWORD *)(a1 + 708) = *(_DWORD *)(a1 + 644);
  *(_QWORD *)(a1 + 712) = *(_QWORD *)(a1 + 624);
  *(_DWORD *)(a1 + 648) = *(_DWORD *)(a1 + 748);
  if ( v3 > *(_DWORD *)(a1 + 128) / 0xAu )
  {
    if ( v3 <= *(_DWORD *)(a1 + 128) >> 1 )
    {
      v4 -= *(_DWORD *)(a1 + 752) / 2;
      *(_DWORD *)(a1 + 760) = v4;
    }
  }
  else
  {
    v4 -= *(_DWORD *)(a1 + 752);
    *(_DWORD *)(a1 + 760) = v4;
  }
  *(_DWORD *)(a1 + 744) = v4;
  return 0;
}
