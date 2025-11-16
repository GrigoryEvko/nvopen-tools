// Function: sub_8F0F10
// Address: 0x8f0f10
//
void *__fastcall sub_8F0F10(_DWORD *a1, __int64 a2)
{
  int v2; // r12d
  int v3; // eax
  __int64 v4; // r13
  int v5; // eax
  int v6; // edx
  int v7; // eax
  __int64 *p_dest; // r14
  int v9; // eax
  int v10; // r15d
  unsigned __int8 *v11; // rcx
  int v12; // ecx
  __int64 v13; // r9
  __int128 *v14; // rdi
  __int64 v15; // r15
  int v16; // eax
  int v17; // r12d
  int v18; // eax
  char *v19; // rsi
  unsigned __int8 *v20; // r15
  int v21; // r14d
  int v22; // r12d
  int v23; // eax
  __int64 v24; // r12
  int v25; // r13d
  int v26; // eax
  int v27; // ebx
  unsigned int v28; // eax
  int v29; // edx
  int v30; // eax
  int v31; // edx
  __int64 v32; // rt2
  int v33; // ebx
  __int128 *v34; // rax
  int v35; // r11d
  int v36; // r10d
  __int128 *v37; // r8
  int v38; // eax
  bool v39; // sf
  int v40; // r11d
  char *v41; // r8
  __int64 v42; // rax
  unsigned __int8 v43; // dl
  __int64 v44; // rax
  int v45; // r11d
  int v46; // edx
  int v47; // edx
  int v48; // eax
  int v49; // r9d
  __int64 v51; // rdx
  int v52; // r11d
  int v53; // eax
  int v54; // eax
  int v55; // esi
  __int64 v56; // rsi
  int v57; // ecx
  int v58; // eax
  unsigned __int8 *v59; // rsi
  int v60; // r9d
  int v61; // eax
  int v62; // edi
  int v63; // r9d
  int v64; // edx
  int v65; // eax
  int v66; // edi
  int v67; // edi
  __int64 v68; // rsi
  int v69; // eax
  int v70; // eax
  int v72; // [rsp+10h] [rbp-E0h]
  int v73; // [rsp+14h] [rbp-DCh]
  int v75; // [rsp+20h] [rbp-D0h]
  int v76; // [rsp+24h] [rbp-CCh]
  __int64 v77; // [rsp+28h] [rbp-C8h]
  size_t n; // [rsp+30h] [rbp-C0h]
  __int64 v79; // [rsp+48h] [rbp-A8h]
  int v80; // [rsp+58h] [rbp-98h]
  int v81; // [rsp+5Ch] [rbp-94h]
  __int64 dest; // [rsp+61h] [rbp-8Fh] BYREF
  int v83; // [rsp+69h] [rbp-87h]
  __int16 v84; // [rsp+6Dh] [rbp-83h]
  char v85; // [rsp+6Fh] [rbp-81h]
  __int128 v86; // [rsp+70h] [rbp-80h] BYREF
  __int128 v87; // [rsp+80h] [rbp-70h] BYREF
  char v88; // [rsp+90h] [rbp-60h]
  _OWORD v89[5]; // [rsp+A0h] [rbp-50h] BYREF

  v2 = *(_DWORD *)(a2 + 28);
  v88 = 0;
  v3 = v2 + 14;
  v89[0] = 0;
  if ( v2 + 7 >= 0 )
    v3 = v2 + 7;
  v73 = v2;
  dest = 0;
  v84 = 0;
  v4 = v3 >> 3;
  v85 = 0;
  v72 = v4;
  v80 = 8 * v4 + 16;
  v5 = a1[7];
  v81 = v4 + 1;
  v6 = v5 + 14;
  v7 = v5 + 7;
  if ( v7 < 0 )
    v7 = v6;
  v89[1] = 0;
  v83 = 0;
  p_dest = (__int64 *)(a2 + 12);
  v9 = v7 >> 3;
  v87 = 0;
  v10 = v9 + 1;
  memcpy((char *)v89 + (int)v4 + 1, a1 + 3, v9);
  v75 = v4 + v10;
  if ( (v2 & 7) != 0 )
  {
    if ( v2 > 0 )
      memcpy(&dest, p_dest, (unsigned int)(v4 - 1) + 1LL);
    p_dest = &dest;
    sub_8EE880(v89, 8 * v75, 8 - v2 % 8);
    sub_8EE880(&dest, 8 * v4, 8 - *(_DWORD *)(a2 + 28) % 8);
  }
  v11 = (unsigned __int8 *)p_dest + v4 - 1;
  v76 = 256 / (*v11 + 1);
  if ( v76 == 1 )
    goto LABEL_10;
  v59 = (unsigned __int8 *)v89;
  v60 = v75 + 13;
  if ( v75 + 6 >= 0 )
    v60 = v75 + 6;
  v61 = 0;
  v62 = 0;
  v63 = v60 >> 3;
  if ( v75 > 1 )
  {
    do
    {
      v64 = *v59;
      ++v62;
      ++v59;
      v65 = v76 * v64 + v61;
      *(v59 - 1) = v65;
      v61 = v65 >> 8;
    }
    while ( v62 < v63 );
  }
  v66 = v4 + 14;
  if ( (int)v4 + 7 >= 0 )
    v66 = v4 + 7;
  v67 = v66 >> 3;
  if ( v2 <= 0 )
  {
LABEL_10:
    if ( v10 < 0 )
      goto LABEL_34;
  }
  else
  {
    v68 = 0;
    v69 = 0;
    do
    {
      v70 = v76 * *((unsigned __int8 *)p_dest + v68) + v69;
      *((_BYTE *)p_dest + v68++) = v70;
      v69 = v70 >> 8;
    }
    while ( v67 > (int)v68 );
    if ( v10 < 0 )
      goto LABEL_34;
  }
  v12 = *v11;
  v13 = v10;
  v14 = &v86;
  v15 = v4 + v10;
  v16 = 8 * v4 + 14;
  v17 = 8 * v4 + 22;
  if ( 8 * (int)v4 + 7 >= 0 )
    v16 = 8 * v4 + 7;
  v18 = v16 >> 3;
  if ( 8 * (int)v4 + 15 >= 0 )
    v17 = 8 * v4 + 15;
  v19 = (char *)p_dest;
  v20 = (unsigned __int8 *)v89 + v15;
  n = v18;
  v21 = v12;
  v22 = v17 >> 3;
  v77 = v22;
  v23 = v22;
  v24 = v4;
  v25 = v23;
  do
  {
    v26 = *v20;
    v27 = *(v20 - 1);
    v86 = 0;
    v29 = (v27 + (v26 << 8)) >> 31;
    v28 = v27 + (v26 << 8);
    v32 = __SPAIR64__(v29, v28) % v21;
    v30 = __SPAIR64__(v29, v28) / v21;
    v31 = v32;
    v33 = v30;
    do
    {
      if ( v33 != 256 && *(v20 - 2) + (v31 << 8) >= v33 * (unsigned __int8)v19[v24 - 2] )
        break;
      v31 += v21;
      --v33;
    }
    while ( v31 <= 255 );
    v79 = v13;
    v34 = (__int128 *)memcpy(v14, v19, n);
    v35 = 0;
    v36 = 0;
    v14 = v34;
    v37 = v34;
    if ( v81 > 0 )
    {
      do
      {
        v38 = v33 * *(unsigned __int8 *)v37;
        v39 = v38 + v35 < 0;
        v40 = v38 + v35;
        *(_BYTE *)v37 = v40;
        if ( v39 )
          v40 += 255;
        ++v36;
        v37 = (__int128 *)((char *)v37 + 1);
        v35 = v40 >> 8;
      }
      while ( v36 < v25 );
    }
    v41 = (char *)v89 + v79;
    v42 = v77;
    do
    {
      if ( (int)v42 <= 0 )
        goto LABEL_27;
      v43 = v41[--v42];
    }
    while ( v43 == *((_BYTE *)v14 + v42) );
    if ( v43 >= *((_BYTE *)v14 + v42) )
    {
LABEL_27:
      if ( v81 <= 0 )
        goto LABEL_32;
      goto LABEL_28;
    }
    LOBYTE(v33) = v33 - 1;
    if ( v81 <= 0 )
      goto LABEL_32;
    v51 = 0;
    v52 = 0;
    do
    {
      v53 = v52 + *((unsigned __int8 *)v14 + v51);
      v52 = 0;
      v54 = v53 - (unsigned __int8)v19[v51];
      if ( v54 < 0 )
        v52 = -1;
      *((_BYTE *)v14 + v51++) = v54;
    }
    while ( v25 > (int)v51 );
LABEL_28:
    v44 = 0;
    v45 = 0;
    do
    {
      v46 = v45 + (unsigned __int8)v41[v44];
      v45 = 0;
      v47 = v46 - *((unsigned __int8 *)v14 + v44);
      if ( v47 < 0 )
        v45 = -1;
      v41[v44++] = v47;
    }
    while ( v25 > (int)v44 );
LABEL_32:
    --v20;
    *((_BYTE *)&v87 + v79) = v33;
    v13 = v79 - 1;
  }
  while ( (int)v79 - 1 >= 0 );
  p_dest = (__int64 *)v19;
LABEL_34:
  v48 = sub_8EE4D0(v89, 8 * v75 - 8);
  if ( v49 != v48 && (v87 & 0x7F) == 0 )
    LOBYTE(v87) = v87 + 1;
  if ( v76 != 1 )
  {
    v55 = v72 + 14;
    if ( v72 + 7 >= 0 )
      v55 = v72 + 7;
    LODWORD(v56) = v55 >> 3;
    if ( v73 > 0 )
    {
      v56 = (int)v56;
      v57 = 0;
      do
      {
        v58 = *((unsigned __int8 *)p_dest + v56 - 1) + v57;
        v57 = v58 << 8;
        *((_BYTE *)p_dest + --v56) = v58 / v76;
      }
      while ( (int)v56 > 0 );
    }
  }
  a1[2] = a1[2] - *(_DWORD *)(a2 + 8) + 8;
  return sub_8EF4C0(a1, (char *)&v87, v80);
}
