// Function: sub_F566B0
// Address: 0xf566b0
//
__int64 __fastcall sub_F566B0(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // r9
  int v10; // edx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rax
  int v16; // r14d
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // edx
  char *v20; // rax
  __int64 v21; // r8
  int v22; // ecx
  __int64 v23; // rdx
  unsigned __int8 *v24; // r14
  __int64 v25; // rdx
  __int64 v26; // r13
  const char *v27; // rax
  _BYTE *v28; // r15
  unsigned __int64 v29; // rdx
  _BYTE *v30; // rdi
  _BYTE *v31; // rdx
  int v32; // esi
  __int64 v33; // rax
  __int64 v34; // rsi
  _QWORD *v35; // rax
  __int64 v36; // r13
  __int64 v37; // r11
  __int64 v38; // r15
  __int64 v39; // rsi
  __int64 *v40; // r8
  __int64 v41; // rdx
  __int64 v42; // rsi
  _QWORD *v43; // rdi
  _QWORD *v44; // rbx
  _QWORD *v45; // r12
  __int64 v46; // rdi
  __int64 v48; // rsi
  unsigned __int8 *v49; // rsi
  __int64 v50; // [rsp+0h] [rbp-180h]
  unsigned __int16 v52; // [rsp+1Eh] [rbp-162h]
  __int64 v53; // [rsp+20h] [rbp-160h]
  __int64 *v54; // [rsp+20h] [rbp-160h]
  __int64 v55; // [rsp+28h] [rbp-158h]
  __int64 v56; // [rsp+30h] [rbp-150h]
  __int64 v57; // [rsp+38h] [rbp-148h]
  __int64 v59; // [rsp+50h] [rbp-130h]
  __int64 v60; // [rsp+58h] [rbp-128h]
  __int64 *v61; // [rsp+60h] [rbp-120h]
  __int64 v62; // [rsp+60h] [rbp-120h]
  __int64 *v63; // [rsp+60h] [rbp-120h]
  __int64 v64; // [rsp+60h] [rbp-120h]
  __int64 v65; // [rsp+70h] [rbp-110h] BYREF
  unsigned __int16 v66; // [rsp+78h] [rbp-108h]
  unsigned __int8 *v67; // [rsp+80h] [rbp-100h] BYREF
  unsigned __int64 v68; // [rsp+88h] [rbp-F8h]
  __int16 v69; // [rsp+A0h] [rbp-E0h]
  _BYTE *v70; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v71; // [rsp+B8h] [rbp-C8h]
  _BYTE v72[64]; // [rsp+C0h] [rbp-C0h] BYREF
  const char *v73; // [rsp+100h] [rbp-80h] BYREF
  __int64 v74; // [rsp+108h] [rbp-78h]
  _QWORD v75[2]; // [rsp+110h] [rbp-70h] BYREF
  __int16 v76; // [rsp+120h] [rbp-60h]

  v5 = *((_QWORD *)a1 + 5);
  v76 = 773;
  v73 = sub_BD5D20((__int64)a1);
  v74 = v6;
  v75[0] = ".noexc";
  v60 = sub_F36990(v5, (__int64 *)a1 + 3, 0, a3, 0, 0, (void **)&v73, 0);
  v7 = *(_QWORD *)(v5 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v8 = (_QWORD *)(v7 - 24);
  if ( !v7 )
    v8 = 0;
  sub_B43D60(v8);
  v10 = *a1;
  if ( v10 == 40 )
  {
    v11 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v11 = -32;
    if ( v10 != 85 )
    {
      v11 = -96;
      if ( v10 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) != 0 )
  {
    v12 = sub_BD2BC0((__int64)a1);
    v14 = v12 + v13;
    v15 = 0;
    if ( (a1[7] & 0x80u) != 0 )
      v15 = sub_BD2BC0((__int64)a1);
    if ( (unsigned int)((v14 - v15) >> 4) )
    {
      if ( (a1[7] & 0x80u) == 0 )
        BUG();
      v16 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
      if ( (a1[7] & 0x80u) == 0 )
        BUG();
      v17 = sub_BD2BC0((__int64)a1);
      v11 -= 32LL * (unsigned int)(*(_DWORD *)(v17 + v18 - 4) - v16);
    }
  }
  v19 = *((_DWORD *)a1 + 1);
  v20 = (char *)v75;
  v21 = (__int64)&a1[v11];
  v74 = 0x800000000LL;
  v22 = 0;
  v23 = 32LL * (v19 & 0x7FFFFFF);
  v73 = (const char *)v75;
  v24 = &a1[-v23];
  v25 = v11 + v23;
  v26 = v25 >> 5;
  if ( (unsigned __int64)v25 > 0x100 )
  {
    v64 = v21;
    sub_C8D5F0((__int64)&v73, v75, v25 >> 5, 8u, v21, v9);
    v22 = v74;
    v21 = v64;
    v20 = (char *)&v73[8 * (unsigned int)v74];
  }
  if ( v24 != (unsigned __int8 *)v21 )
  {
    do
    {
      if ( v20 )
        *(_QWORD *)v20 = *(_QWORD *)v24;
      v24 += 32;
      v20 += 8;
    }
    while ( (unsigned __int8 *)v21 != v24 );
    v22 = v74;
  }
  v70 = v72;
  LODWORD(v74) = v26 + v22;
  v71 = 0x100000000LL;
  sub_B56970((__int64)a1, (__int64)&v70);
  sub_B43C20((__int64)&v65, v5);
  v27 = sub_BD5D20((__int64)a1);
  v28 = v70;
  v67 = (unsigned __int8 *)v27;
  v69 = 261;
  v61 = (__int64 *)v73;
  v68 = v29;
  v53 = (unsigned int)v74;
  v57 = *((_QWORD *)a1 - 4);
  v59 = *((_QWORD *)a1 + 10);
  v56 = v65;
  v52 = v66;
  v30 = &v70[56 * (unsigned int)v71];
  if ( v70 == v30 )
  {
    v32 = 0;
  }
  else
  {
    v31 = v70;
    v32 = 0;
    do
    {
      v33 = *((_QWORD *)v31 + 5) - *((_QWORD *)v31 + 4);
      v31 += 56;
      v32 += v33 >> 3;
    }
    while ( v30 != v31 );
  }
  v34 = (unsigned int)(v74 + v32 + 3);
  v50 = (unsigned int)v71;
  LOBYTE(v24) = 16 * (_DWORD)v71 != 0;
  v35 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v71) << 32) | v34);
  v36 = (__int64)v35;
  if ( v35 )
  {
    v37 = v53;
    v54 = v61;
    v55 = v37;
    v62 = (__int64)v28;
    v38 = (__int64)v35;
    sub_B44260((__int64)v35, **(_QWORD **)(v59 + 16), 5, v34 & 0x7FFFFFF | ((_DWORD)v24 << 28), v56, v52);
    *(_QWORD *)(v36 + 72) = 0;
    sub_B4A9C0(v36, v59, v57, v60, a2, (__int64)&v67, v54, v55, v62, v50);
  }
  else
  {
    v38 = 0;
  }
  v39 = *((_QWORD *)a1 + 6);
  v40 = (__int64 *)(v36 + 48);
  v67 = (unsigned __int8 *)v39;
  if ( !v39 )
  {
    if ( v40 == (__int64 *)&v67 )
      goto LABEL_28;
    v48 = *(_QWORD *)(v36 + 48);
    if ( !v48 )
      goto LABEL_28;
LABEL_49:
    v63 = v40;
    sub_B91220((__int64)v40, v48);
    v40 = v63;
    goto LABEL_50;
  }
  sub_B96E90((__int64)&v67, v39, 1);
  v40 = (__int64 *)(v36 + 48);
  if ( (unsigned __int8 **)(v36 + 48) == &v67 )
  {
    if ( v67 )
      sub_B91220((__int64)&v67, (__int64)v67);
    goto LABEL_28;
  }
  v48 = *(_QWORD *)(v36 + 48);
  if ( v48 )
    goto LABEL_49;
LABEL_50:
  v49 = v67;
  *(_QWORD *)(v36 + 48) = v67;
  if ( v49 )
    sub_B976B0((__int64)&v67, v49, (__int64)v40);
LABEL_28:
  v41 = 0;
  *(_WORD *)(v36 + 2) = *((_WORD *)a1 + 1) & 0xFFC | *(_WORD *)(v36 + 2) & 0xF003;
  *(_QWORD *)(v36 + 72) = *((_QWORD *)a1 + 9);
  if ( (a1[7] & 0x20) != 0 )
    v41 = sub_B91C10((__int64)a1, 2);
  sub_B99FD0(v38, 2u, v41);
  if ( a3 )
  {
    v67 = (unsigned __int8 *)v5;
    v68 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    sub_FFB3D0(a3, &v67, 1);
  }
  v42 = v36;
  sub_BD84D0((__int64)a1, v36);
  v43 = *(_QWORD **)(v60 + 56);
  if ( v43 )
    v43 -= 3;
  sub_B43D60(v43);
  v44 = v70;
  v45 = &v70[56 * (unsigned int)v71];
  if ( v70 != (_BYTE *)v45 )
  {
    do
    {
      v46 = *(v45 - 3);
      v45 -= 7;
      if ( v46 )
      {
        v42 = v45[6] - v46;
        j_j___libc_free_0(v46, v42);
      }
      if ( (_QWORD *)*v45 != v45 + 2 )
      {
        v42 = v45[2] + 1LL;
        j_j___libc_free_0(*v45, v42);
      }
    }
    while ( v44 != v45 );
    v45 = v70;
  }
  if ( v45 != (_QWORD *)v72 )
    _libc_free(v45, v42);
  if ( v73 != (const char *)v75 )
    _libc_free(v73, v42);
  return v60;
}
