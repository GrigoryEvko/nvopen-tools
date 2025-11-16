// Function: sub_2D91E40
// Address: 0x2d91e40
//
__int64 __fastcall sub_2D91E40(__int64 a1, _DWORD *a2)
{
  __int16 v4; // ax
  __int64 v5; // rax
  __int64 v6; // rax
  int *v7; // rdi
  char v8; // al
  int v9; // eax
  _BOOL4 v10; // edx
  bool v11; // al
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned __int16 v14; // r12
  int v15; // edx
  bool v16; // al
  int v17; // ecx
  __int16 v18; // ax
  int v19; // ecx
  unsigned __int16 v20; // dx
  bool v21; // al
  size_t v22; // rsi
  int v23; // edx
  _QWORD *v24; // rdi
  __int64 v25; // rcx
  __int64 v26; // rdx
  _QWORD *v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // rdx
  _QWORD *v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rdx
  _QWORD *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rdx
  _QWORD *v36; // rdi
  __int64 v37; // rcx
  __int64 v38; // rdx
  _QWORD *v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // rdx
  unsigned __int64 v42; // r12
  unsigned __int64 *v43; // rbx
  unsigned __int64 *v44; // rax
  unsigned __int64 *v45; // r13
  unsigned __int64 *v46; // rax
  __int64 v47; // rax
  unsigned __int64 *v48; // rbx
  unsigned __int64 *v49; // r12
  unsigned __int64 v50; // rdi
  size_t v52; // rdx
  size_t v53; // rdx
  size_t v54; // rdx
  size_t v55; // rdx
  size_t v56; // rdx
  size_t v57; // rdx
  unsigned int v58; // r13d
  __int64 v59; // [rsp+0h] [rbp-160h]
  int v60; // [rsp+30h] [rbp-130h] BYREF
  __int16 v61; // [rsp+34h] [rbp-12Ch]
  __int64 v62; // [rsp+38h] [rbp-128h]
  __int64 v63; // [rsp+40h] [rbp-120h]
  __int64 v64; // [rsp+48h] [rbp-118h]
  _QWORD *v65; // [rsp+50h] [rbp-110h]
  size_t v66; // [rsp+58h] [rbp-108h]
  _QWORD v67[2]; // [rsp+60h] [rbp-100h] BYREF
  _QWORD *v68; // [rsp+70h] [rbp-F0h]
  size_t v69; // [rsp+78h] [rbp-E8h]
  _QWORD v70[2]; // [rsp+80h] [rbp-E0h] BYREF
  _QWORD *v71; // [rsp+90h] [rbp-D0h]
  size_t v72; // [rsp+98h] [rbp-C8h]
  _QWORD v73[2]; // [rsp+A0h] [rbp-C0h] BYREF
  _QWORD *v74; // [rsp+B0h] [rbp-B0h]
  size_t v75; // [rsp+B8h] [rbp-A8h]
  _QWORD v76[2]; // [rsp+C0h] [rbp-A0h] BYREF
  _QWORD *v77; // [rsp+D0h] [rbp-90h]
  size_t v78; // [rsp+D8h] [rbp-88h]
  _QWORD v79[2]; // [rsp+E0h] [rbp-80h] BYREF
  _QWORD *v80; // [rsp+F0h] [rbp-70h]
  size_t n; // [rsp+F8h] [rbp-68h]
  _QWORD src[2]; // [rsp+100h] [rbp-60h] BYREF
  unsigned __int64 *v83; // [rsp+110h] [rbp-50h]
  unsigned __int64 *v84; // [rsp+118h] [rbp-48h]
  __int64 v85; // [rsp+120h] [rbp-40h]
  char v86; // [rsp+128h] [rbp-38h]

  v4 = *(_WORD *)(a1 + 8);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_WORD *)(a1 + 8) = v4 & 0xE000 | 0x408;
  *(_QWORD *)(a1 + 12) = 0x100000001LL;
  v5 = *(_QWORD *)(a1 + 20);
  *(_BYTE *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 1;
  *(_QWORD *)(a1 + 20) = (unsigned int)v5 & 0xE0000000 | 0x300000060LL;
  LOWORD(v5) = *(_WORD *)(a1 + 48);
  *(_QWORD *)(a1 + 104) = 1;
  *(_WORD *)(a1 + 48) = v5 & 0x8000 | 0x4020;
  v6 = a1 + 72;
  v7 = (int *)(a1 + 120);
  *((_QWORD *)v7 - 8) = v6;
  *((_QWORD *)v7 - 1) = 4294901760LL;
  sub_EA1890(v7);
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 376) = a1 + 392;
  *(_BYTE *)(a1 + 392) = 0;
  *(_DWORD *)(a1 + 96) = sub_2D91410();
  *(_BYTE *)(a1 + 8) = sub_2D91360() & 1 | *(_BYTE *)(a1 + 8) & 0xFE;
  *(_BYTE *)(a1 + 8) = (2 * (sub_2D91370() & 1)) | *(_BYTE *)(a1 + 8) & 0xFD;
  *(_BYTE *)(a1 + 8) = (4 * (sub_2D91380() & 1)) | *(_BYTE *)(a1 + 8) & 0xFB;
  *(_BYTE *)(a1 + 8) = (16 * (sub_2D91390() & 1)) | *(_BYTE *)(a1 + 8) & 0xEF;
  *(_BYTE *)(a1 + 8) = (32 * (sub_2D913A0() & 1)) | *(_BYTE *)(a1 + 8) & 0xDF;
  *(_BYTE *)(a1 + 8) = (8 * (sub_2D913B0() & 1)) | *(_BYTE *)(a1 + 8) & 0xF7;
  v8 = sub_2D913D0();
  *(_BYTE *)(a1 + 112) = v8;
  *(_BYTE *)(a1 + 113) = v8;
  *(_BYTE *)(a1 + 8) = ((unsigned __int8)sub_2D913F0() << 7) | *(_BYTE *)(a1 + 8) & 0x7F;
  if ( (unsigned int)sub_2D91400() )
    *(_DWORD *)(a1 + 92) = sub_2D91400();
  *(_BYTE *)(a1 + 8) = ((sub_2D913C0() & 1) << 6) | *(_BYTE *)(a1 + 8) & 0xBF;
  *(_BYTE *)(a1 + 9) = sub_2D91430() & 1 | *(_BYTE *)(a1 + 9) & 0xFE;
  *(_BYTE *)(a1 + 9) = (2 * (sub_2D91440() & 1)) | *(_BYTE *)(a1 + 9) & 0xFD;
  *(_BYTE *)(a1 + 9) = (4 * (sub_2D91460() & 1)) | *(_BYTE *)(a1 + 9) & 0xFB;
  *(_BYTE *)(a1 + 20) = (sub_2D91540() ^ 1) & 1 | *(_BYTE *)(a1 + 20) & 0xFE;
  *(_BYTE *)(a1 + 20) = (2 * (sub_2D91550() & 1)) | *(_BYTE *)(a1 + 20) & 0xFD;
  LOWORD(v9) = sub_2D91560();
  v10 = v9;
  v11 = 1;
  if ( a2[13] != 8 )
    v11 = (unsigned int)(a2[8] - 56) <= 1;
  v12 = v10;
  LOWORD(v12) = BYTE1(v10);
  if ( BYTE1(v10) )
    v11 = v10;
  v13 = *(_BYTE *)(a1 + 20) & 0xF7;
  *(_BYTE *)(a1 + 20) = *(_BYTE *)(a1 + 20) & 0xF7 | (8 * v11);
  *(_BYTE *)(a1 + 20) = (4 * (sub_2D91660(v7, a2, v13, v12) & 1)) | *(_BYTE *)(a1 + 20) & 0xFB;
  *(_BYTE *)(a1 + 20) = (16 * (sub_2D91670() & 1)) | *(_BYTE *)(a1 + 20) & 0xEF;
  *(_BYTE *)(a1 + 20) = (32 * (sub_2D91680() & 1)) | *(_BYTE *)(a1 + 20) & 0xDF;
  *(_BYTE *)(a1 + 23) = (16 * (sub_2D91690() & 1)) | *(_BYTE *)(a1 + 23) & 0xEF;
  *(_DWORD *)(a1 + 24) = sub_2D91A80(a1);
  *(_BYTE *)(a1 + 20) = ((sub_2D91970() & 1) << 6) | *(_BYTE *)(a1 + 20) & 0xBF;
  *(_BYTE *)(a1 + 20) = ((unsigned __int8)sub_2D91980() << 7) | *(_BYTE *)(a1 + 20) & 0x7F;
  *(_BYTE *)(a1 + 21) = sub_2D91990() & 1 | *(_BYTE *)(a1 + 21) & 0xFE;
  *(_DWORD *)(a1 + 20) = ((unsigned __int8)sub_2D91760() << 11) | *(_DWORD *)(a1 + 20) & 0xFFF807FF;
  v14 = sub_2D91770();
  if ( a2[12] != 17
    || ((v58 = sub_CC7810((__int64)a2), !(v16 = sub_CC7F40((__int64)a2))) || v58 > 0x14) && (v16 = 1, v58 > 0x1C) )
  {
    v15 = a2[11];
    v16 = 1;
    if ( v15 != 11 )
    {
      v17 = a2[12];
      if ( v15 != 14 || v17 != 29 )
        v16 = v17 == 49 || v15 == 39;
    }
  }
  if ( HIBYTE(v14) )
    v16 = v14;
  *(_BYTE *)(a1 + 22) = *(_BYTE *)(a1 + 22) & 0xF7 | (8 * v16);
  v18 = sub_2D91870();
  v19 = a2[8];
  v20 = v18;
  v21 = 1;
  v22 = (unsigned int)(v19 - 3);
  if ( (unsigned int)v22 > 2 )
  {
    v21 = v19 == 29 && a2[12] == 17;
    if ( !v21 )
      v21 = a2[11] == 4;
  }
  if ( HIBYTE(v20) )
    v21 = v20;
  *(_BYTE *)(a1 + 22) = *(_BYTE *)(a1 + 22) & 0xEF | (16 * v21);
  *(_DWORD *)(a1 + 116) = sub_2D91340();
  *(_BYTE *)(a1 + 22) = ((sub_2D919C0() & 1) << 6) | *(_BYTE *)(a1 + 22) & 0xBF;
  *(_BYTE *)(a1 + 23) = sub_2D919F0() & 1 | *(_BYTE *)(a1 + 23) & 0xFE;
  *(_BYTE *)(a1 + 23) = (2 * (sub_2D91A00() & 1)) | *(_BYTE *)(a1 + 23) & 0xFD;
  *(_BYTE *)(a1 + 23) = (8 * (sub_2D919D0() & 1)) | *(_BYTE *)(a1 + 23) & 0xF7;
  *(_BYTE *)(a1 + 48) = sub_2D919E0() & 1 | *(_BYTE *)(a1 + 48) & 0xFE;
  *(_BYTE *)(a1 + 48) = (4 * (sub_2D91A10() & 1)) | *(_BYTE *)(a1 + 48) & 0xFB;
  *(_BYTE *)(a1 + 48) = (16 * (sub_2D91A20() & 1)) | *(_BYTE *)(a1 + 48) & 0xEF;
  *(_BYTE *)(a1 + 48) = (32 * (sub_2D91A30() & 1)) | *(_BYTE *)(a1 + 48) & 0xDF;
  *(_BYTE *)(a1 + 48) = ((sub_2D91A40() & 1) << 6) | *(_BYTE *)(a1 + 48) & 0xBF;
  *(_DWORD *)(a1 + 88) = sub_2D91A50();
  *(_BYTE *)(a1 + 49) = (2 * (sub_2D91A60() & 1)) | *(_BYTE *)(a1 + 49) & 0xFD;
  *(_BYTE *)(a1 + 49) = (16 * (sub_2D91A70() & 1)) | *(_BYTE *)(a1 + 49) & 0xEF;
  sub_3148A20(&v60);
  v23 = v60;
  v24 = *(_QWORD **)(a1 + 152);
  BYTE1(v23) = BYTE1(v60) & 0x3F;
  *(_DWORD *)(a1 + 120) = v23 | *(_DWORD *)(a1 + 120) & 0xC000;
  *(_WORD *)(a1 + 124) = v61;
  *(_QWORD *)(a1 + 128) = v62;
  *(_QWORD *)(a1 + 136) = v63;
  *(_QWORD *)(a1 + 144) = v64;
  if ( v65 == v67 )
  {
    v55 = v66;
    if ( v66 )
    {
      if ( v66 == 1 )
      {
        *(_BYTE *)v24 = v67[0];
      }
      else
      {
        v22 = (size_t)v67;
        memcpy(v24, v67, v66);
      }
      v55 = v66;
      v24 = *(_QWORD **)(a1 + 152);
    }
    *(_QWORD *)(a1 + 160) = v55;
    *((_BYTE *)v24 + v55) = 0;
    v24 = v65;
  }
  else
  {
    v22 = v66;
    v25 = v67[0];
    if ( v24 == (_QWORD *)(a1 + 168) )
    {
      *(_QWORD *)(a1 + 152) = v65;
      *(_QWORD *)(a1 + 160) = v22;
      *(_QWORD *)(a1 + 168) = v25;
    }
    else
    {
      v26 = *(_QWORD *)(a1 + 168);
      *(_QWORD *)(a1 + 152) = v65;
      *(_QWORD *)(a1 + 160) = v22;
      *(_QWORD *)(a1 + 168) = v25;
      if ( v24 )
      {
        v65 = v24;
        v67[0] = v26;
        goto LABEL_20;
      }
    }
    v65 = v67;
    v24 = v67;
  }
LABEL_20:
  v66 = 0;
  *(_BYTE *)v24 = 0;
  v27 = *(_QWORD **)(a1 + 184);
  if ( v68 == v70 )
  {
    v54 = v69;
    if ( v69 )
    {
      if ( v69 == 1 )
      {
        *(_BYTE *)v27 = v70[0];
      }
      else
      {
        v22 = (size_t)v70;
        memcpy(v27, v70, v69);
      }
      v54 = v69;
      v27 = *(_QWORD **)(a1 + 184);
    }
    *(_QWORD *)(a1 + 192) = v54;
    *((_BYTE *)v27 + v54) = 0;
    v27 = v68;
  }
  else
  {
    v22 = v69;
    v28 = v70[0];
    if ( v27 == (_QWORD *)(a1 + 200) )
    {
      *(_QWORD *)(a1 + 184) = v68;
      *(_QWORD *)(a1 + 192) = v22;
      *(_QWORD *)(a1 + 200) = v28;
    }
    else
    {
      v29 = *(_QWORD *)(a1 + 200);
      *(_QWORD *)(a1 + 184) = v68;
      *(_QWORD *)(a1 + 192) = v22;
      *(_QWORD *)(a1 + 200) = v28;
      if ( v27 )
      {
        v68 = v27;
        v70[0] = v29;
        goto LABEL_24;
      }
    }
    v68 = v70;
    v27 = v70;
  }
LABEL_24:
  v69 = 0;
  *(_BYTE *)v27 = 0;
  v30 = *(_QWORD **)(a1 + 216);
  if ( v71 == v73 )
  {
    v57 = v72;
    if ( v72 )
    {
      if ( v72 == 1 )
      {
        *(_BYTE *)v30 = v73[0];
      }
      else
      {
        v22 = (size_t)v73;
        memcpy(v30, v73, v72);
      }
      v57 = v72;
      v30 = *(_QWORD **)(a1 + 216);
    }
    *(_QWORD *)(a1 + 224) = v57;
    *((_BYTE *)v30 + v57) = 0;
    v30 = v71;
  }
  else
  {
    v22 = v72;
    v31 = v73[0];
    if ( v30 == (_QWORD *)(a1 + 232) )
    {
      *(_QWORD *)(a1 + 216) = v71;
      *(_QWORD *)(a1 + 224) = v22;
      *(_QWORD *)(a1 + 232) = v31;
    }
    else
    {
      v32 = *(_QWORD *)(a1 + 232);
      *(_QWORD *)(a1 + 216) = v71;
      *(_QWORD *)(a1 + 224) = v22;
      *(_QWORD *)(a1 + 232) = v31;
      if ( v30 )
      {
        v71 = v30;
        v73[0] = v32;
        goto LABEL_28;
      }
    }
    v71 = v73;
    v30 = v73;
  }
LABEL_28:
  v72 = 0;
  *(_BYTE *)v30 = 0;
  v33 = *(_QWORD **)(a1 + 248);
  if ( v74 == v76 )
  {
    v56 = v75;
    if ( v75 )
    {
      if ( v75 == 1 )
      {
        *(_BYTE *)v33 = v76[0];
      }
      else
      {
        v22 = (size_t)v76;
        memcpy(v33, v76, v75);
      }
      v56 = v75;
      v33 = *(_QWORD **)(a1 + 248);
    }
    *(_QWORD *)(a1 + 256) = v56;
    *((_BYTE *)v33 + v56) = 0;
    v33 = v74;
  }
  else
  {
    v22 = v75;
    v34 = v76[0];
    if ( v33 == (_QWORD *)(a1 + 264) )
    {
      *(_QWORD *)(a1 + 248) = v74;
      *(_QWORD *)(a1 + 256) = v22;
      *(_QWORD *)(a1 + 264) = v34;
    }
    else
    {
      v35 = *(_QWORD *)(a1 + 264);
      *(_QWORD *)(a1 + 248) = v74;
      *(_QWORD *)(a1 + 256) = v22;
      *(_QWORD *)(a1 + 264) = v34;
      if ( v33 )
      {
        v74 = v33;
        v76[0] = v35;
        goto LABEL_32;
      }
    }
    v74 = v76;
    v33 = v76;
  }
LABEL_32:
  v75 = 0;
  *(_BYTE *)v33 = 0;
  v36 = *(_QWORD **)(a1 + 280);
  if ( v77 == v79 )
  {
    v53 = v78;
    if ( v78 )
    {
      if ( v78 == 1 )
      {
        *(_BYTE *)v36 = v79[0];
      }
      else
      {
        v22 = (size_t)v79;
        memcpy(v36, v79, v78);
      }
      v53 = v78;
      v36 = *(_QWORD **)(a1 + 280);
    }
    *(_QWORD *)(a1 + 288) = v53;
    *((_BYTE *)v36 + v53) = 0;
    v36 = v77;
  }
  else
  {
    v22 = v78;
    v37 = v79[0];
    if ( v36 == (_QWORD *)(a1 + 296) )
    {
      *(_QWORD *)(a1 + 280) = v77;
      *(_QWORD *)(a1 + 288) = v22;
      *(_QWORD *)(a1 + 296) = v37;
    }
    else
    {
      v38 = *(_QWORD *)(a1 + 296);
      *(_QWORD *)(a1 + 280) = v77;
      *(_QWORD *)(a1 + 288) = v22;
      *(_QWORD *)(a1 + 296) = v37;
      if ( v36 )
      {
        v77 = v36;
        v79[0] = v38;
        goto LABEL_36;
      }
    }
    v77 = v79;
    v36 = v79;
  }
LABEL_36:
  v78 = 0;
  *(_BYTE *)v36 = 0;
  v39 = *(_QWORD **)(a1 + 312);
  if ( v80 == src )
  {
    v52 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        *(_BYTE *)v39 = src[0];
      }
      else
      {
        v22 = (size_t)src;
        memcpy(v39, src, n);
      }
      v52 = n;
      v39 = *(_QWORD **)(a1 + 312);
    }
    *(_QWORD *)(a1 + 320) = v52;
    *((_BYTE *)v39 + v52) = 0;
    v39 = v80;
  }
  else
  {
    v40 = src[0];
    v22 = n;
    if ( v39 == (_QWORD *)(a1 + 328) )
    {
      *(_QWORD *)(a1 + 312) = v80;
      *(_QWORD *)(a1 + 320) = v22;
      *(_QWORD *)(a1 + 328) = v40;
    }
    else
    {
      v41 = *(_QWORD *)(a1 + 328);
      *(_QWORD *)(a1 + 312) = v80;
      *(_QWORD *)(a1 + 320) = v22;
      *(_QWORD *)(a1 + 328) = v40;
      if ( v39 )
      {
        v80 = v39;
        src[0] = v41;
        goto LABEL_40;
      }
    }
    v80 = src;
    v39 = src;
  }
LABEL_40:
  n = 0;
  *(_BYTE *)v39 = 0;
  v42 = *(_QWORD *)(a1 + 344);
  v43 = *(unsigned __int64 **)(a1 + 352);
  v59 = *(_QWORD *)(a1 + 360);
  v44 = v83;
  v45 = (unsigned __int64 *)v42;
  v83 = 0;
  *(_QWORD *)(a1 + 344) = v44;
  v46 = v84;
  v84 = 0;
  *(_QWORD *)(a1 + 352) = v46;
  v47 = v85;
  v85 = 0;
  for ( *(_QWORD *)(a1 + 360) = v47; v43 != v45; v45 += 4 )
  {
    if ( (unsigned __int64 *)*v45 != v45 + 2 )
    {
      v22 = v45[2] + 1;
      j_j___libc_free_0(*v45);
    }
  }
  if ( v42 )
  {
    v22 = v59 - v42;
    j_j___libc_free_0(v42);
  }
  v48 = v84;
  v49 = v83;
  *(_BYTE *)(a1 + 368) = v86 & 3 | *(_BYTE *)(a1 + 368) & 0xFC;
  if ( v48 != v49 )
  {
    do
    {
      if ( (unsigned __int64 *)*v49 != v49 + 2 )
      {
        v22 = v49[2] + 1;
        j_j___libc_free_0(*v49);
      }
      v49 += 4;
    }
    while ( v48 != v49 );
    v49 = v83;
  }
  if ( v49 )
  {
    v22 = v85 - (_QWORD)v49;
    j_j___libc_free_0((unsigned __int64)v49);
  }
  if ( v80 != src )
  {
    v22 = src[0] + 1LL;
    j_j___libc_free_0((unsigned __int64)v80);
  }
  if ( v77 != v79 )
  {
    v22 = v79[0] + 1LL;
    j_j___libc_free_0((unsigned __int64)v77);
  }
  if ( v74 != v76 )
  {
    v22 = v76[0] + 1LL;
    j_j___libc_free_0((unsigned __int64)v74);
  }
  if ( v71 != v73 )
  {
    v22 = v73[0] + 1LL;
    j_j___libc_free_0((unsigned __int64)v71);
  }
  if ( v68 != v70 )
  {
    v22 = v70[0] + 1LL;
    j_j___libc_free_0((unsigned __int64)v68);
  }
  v50 = (unsigned __int64)v65;
  if ( v65 != v67 )
  {
    v22 = v67[0] + 1LL;
    j_j___libc_free_0((unsigned __int64)v65);
  }
  *(_DWORD *)(a1 + 100) = sub_2D91220(v50, v22);
  *(_DWORD *)(a1 + 104) = sub_2D919A0();
  *(_DWORD *)(a1 + 108) = sub_2D919B0();
  *(_DWORD *)(a1 + 16) = sub_2D91420();
  return a1;
}
