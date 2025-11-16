// Function: sub_2C21A80
// Address: 0x2c21a80
//
__int64 __fastcall sub_2C21A80(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // rax
  __int64 *v4; // rax
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r12
  int v11; // edx
  __int64 *v12; // rax
  int v13; // r14d
  char v14; // r15
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // r11d
  _QWORD *v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned int v24; // r11d
  unsigned __int8 v25; // al
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // r13
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  int v34; // eax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rsi
  _BYTE *v38; // rax
  __int64 v39; // rax
  unsigned int v40; // r11d
  __int64 v41; // rdx
  unsigned __int8 v42; // al
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // [rsp+10h] [rbp-E0h]
  __int64 v47; // [rsp+18h] [rbp-D8h]
  __int64 v48; // [rsp+20h] [rbp-D0h]
  _BYTE *v49; // [rsp+28h] [rbp-C8h]
  __int64 v50; // [rsp+30h] [rbp-C0h]
  __int16 v51; // [rsp+38h] [rbp-B8h]
  char v52; // [rsp+3Bh] [rbp-B5h]
  int v53; // [rsp+3Ch] [rbp-B4h]
  __int64 v54; // [rsp+40h] [rbp-B0h]
  unsigned int v55; // [rsp+48h] [rbp-A8h]
  unsigned int v56; // [rsp+48h] [rbp-A8h]
  unsigned int v57; // [rsp+48h] [rbp-A8h]
  int v58; // [rsp+4Ch] [rbp-A4h]
  __int64 v59; // [rsp+50h] [rbp-A0h]
  __int64 v60; // [rsp+58h] [rbp-98h]
  __int64 v61; // [rsp+60h] [rbp-90h]
  __int64 v62; // [rsp+68h] [rbp-88h]
  __int64 v63; // [rsp+68h] [rbp-88h]
  __int64 v64; // [rsp+70h] [rbp-80h]
  unsigned int v65; // [rsp+70h] [rbp-80h]
  __int64 *v66; // [rsp+70h] [rbp-80h]
  __int64 v68; // [rsp+88h] [rbp-68h]
  __int64 v69[4]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v70; // [rsp+B0h] [rbp-40h]

  v2 = a1;
  v3 = *(_QWORD *)(a2 + 904);
  v54 = v3;
  v53 = *(_DWORD *)(v3 + 104);
  v50 = *(_QWORD *)(v3 + 96);
  v52 = *(_BYTE *)(v3 + 110);
  v51 = *(_WORD *)(v3 + 108);
  if ( *(_BYTE *)(a1 + 152) == 5 )
  {
    v34 = sub_2C1A110(a1);
    v2 = a1;
    *(_DWORD *)(v54 + 104) = v34;
  }
  v4 = *(__int64 **)(v2 + 48);
  BYTE4(v69[0]) = 0;
  LODWORD(v69[0]) = 0;
  v64 = v2;
  v5 = sub_2BFB120(a2, *v4, (unsigned int *)v69);
  BYTE4(v69[0]) = 0;
  v60 = v5;
  v6 = v5;
  v7 = *(_QWORD *)(v64 + 48);
  LODWORD(v69[0]) = 0;
  v8 = sub_2BFB120(a2, *(_QWORD *)(v7 + 8), (unsigned int *)v69);
  v9 = *(_QWORD *)(v6 + 8);
  v10 = *(_QWORD *)(a2 + 904);
  v59 = v8;
  v11 = *(unsigned __int8 *)(v9 + 8);
  if ( (unsigned int)(v11 - 17) <= 1 )
  {
    v12 = *(__int64 **)(v9 + 16);
    v9 = *v12;
    LOBYTE(v11) = *(_BYTE *)(*v12 + 8);
  }
  if ( (_BYTE)v11 == 12 )
  {
    v58 = 17;
    v13 = 13;
  }
  else
  {
    v58 = 18;
    v13 = *(_DWORD *)(v64 + 160);
  }
  v61 = v64 + 96;
  v14 = sub_2C46C30(v64 + 96);
  v15 = sub_BCB060(v9);
  v16 = sub_BCCE00(*(_QWORD **)v9, v15);
  v17 = v64;
  v18 = v16;
  if ( v14 )
  {
    v47 = 0;
    v48 = 0;
    v49 = 0;
    v46 = 0;
    v65 = 1;
  }
  else
  {
    if ( *(_BYTE *)(a2 + 12) )
    {
      v63 = v64;
      v66 = (__int64 *)v16;
      v46 = sub_BCE1B0((__int64 *)v9, *(_QWORD *)(a2 + 8));
      v70 = 257;
      v35 = sub_BCE1B0(v66, *(_QWORD *)(a2 + 8));
      v49 = (_BYTE *)sub_B33FB0(v10, v35, (__int64)v69);
      v70 = 257;
      v48 = sub_B37620((unsigned int **)v10, *(_QWORD *)(a2 + 8), v59, v69);
      v70 = 257;
      v36 = sub_B37620((unsigned int **)v10, *(_QWORD *)(a2 + 8), v60, v69);
      v18 = (__int64)v66;
      v17 = v63;
      v47 = v36;
    }
    else
    {
      v47 = 0;
      v48 = 0;
      v49 = 0;
      v46 = 0;
    }
    v65 = *(_DWORD *)(a2 + 8);
  }
  v19 = 0;
  if ( *(_BYTE *)(a2 + 24) )
  {
    v19 = *(_DWORD *)(a2 + 16);
    v65 = v19 + 1;
  }
  LODWORD(v20) = 0;
  if ( *(_DWORD *)(v17 + 56) == 3 )
  {
    v21 = *(_QWORD *)(*(_QWORD *)(v17 + 48) + 16LL);
    if ( v21 )
    {
      v22 = *(_QWORD *)(v21 + 40);
      v20 = *(_QWORD **)(v22 + 24);
      if ( *(_DWORD *)(v22 + 32) > 0x40u )
        v20 = (_QWORD *)*v20;
    }
  }
  v55 = v19;
  v23 = sub_2AB26E0(v10, v18, *(_QWORD *)(a2 + 8), (int)v20);
  v24 = v55;
  v62 = v23;
  if ( !v14 && *(_BYTE *)(a2 + 12) )
  {
    v37 = *(_QWORD *)(a2 + 8);
    v70 = 257;
    v38 = (_BYTE *)sub_B37620((unsigned int **)v10, v37, v23, v69);
    v70 = 257;
    v39 = sub_929C50((unsigned int **)v10, v38, v49, (__int64)v69, 0, 0);
    v40 = v55;
    v41 = v39;
    v42 = *(_BYTE *)(v9 + 8);
    if ( v42 <= 3u || v42 == 5 || (v42 & 0xFD) == 4 )
    {
      v70 = 257;
      if ( *(_BYTE *)(v10 + 108) )
      {
        BYTE4(v68) = 0;
        v43 = sub_B358C0(v10, 0x88u, v41, v46, v68, (__int64)v69, 0, 0, 0);
      }
      else
      {
        v43 = sub_2C13D90(v10, 44, v41, v46, (__int64)v69, 0, (unsigned int)v68);
      }
      v40 = v55;
      v41 = v43;
    }
    v57 = v40;
    v70 = 257;
    v44 = sub_2C137C0(v10, v58, v41, v48, (unsigned int)v68, (__int64)v69, 0);
    HIDWORD(v68) = 0;
    v70 = 257;
    v45 = sub_2C137C0(v10, v13, v47, v44, (unsigned int)v68, (__int64)v69, 0);
    sub_2BF26E0(a2, v61, v45, 0);
    v24 = v57;
  }
  v25 = *(_BYTE *)(v9 + 8);
  if ( v25 <= 3u || v25 == 5 || (v25 & 0xFD) == 4 )
  {
    v56 = v24;
    v70 = 257;
    if ( *(_BYTE *)(v10 + 108) )
    {
      BYTE4(v68) = 0;
      v26 = sub_B358C0(v10, 0x88u, v62, v9, v68, (__int64)v69, 0, 0, 0);
    }
    else
    {
      v26 = sub_2C13D90(v10, 44, v62, v9, (__int64)v69, 0, (unsigned int)v68);
    }
    v24 = v56;
    v62 = v26;
  }
  v27 = v9;
  v28 = v24;
  if ( v24 < v65 )
  {
    do
    {
      v70 = 257;
      if ( *(_BYTE *)(v27 + 8) == 12 )
        v29 = sub_AD64C0(v27, v28, 1u);
      else
        v29 = (__int64)sub_AD8DD0(v27, (double)(int)v28);
      v30 = sub_2C137C0(v10, v13, v62, v29, (unsigned int)v68, (__int64)v69, 0);
      v70 = 257;
      v31 = sub_2C137C0(v10, v58, v30, v59, (unsigned int)v68, (__int64)v69, 0);
      v70 = 257;
      v32 = sub_2C137C0(v10, v13, v60, v31, (unsigned int)v68, (__int64)v69, 0);
      LODWORD(v69[0]) = v28;
      BYTE4(v69[0]) = 0;
      ++v28;
      sub_2AC6E90(a2, v61, v32, (unsigned int *)v69);
    }
    while ( v65 > (unsigned int)v28 );
  }
  *(_QWORD *)(v54 + 96) = v50;
  *(_BYTE *)(v54 + 110) = v52;
  *(_DWORD *)(v54 + 104) = v53;
  *(_WORD *)(v54 + 108) = v51;
  return v54;
}
