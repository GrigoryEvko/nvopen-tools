// Function: sub_1621740
// Address: 0x1621740
//
__int64 __fastcall sub_1621740(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *v2; // rax
  _QWORD *v3; // rbx
  __int64 result; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // rax
  _QWORD *v8; // rbx
  _QWORD *v9; // rax
  _QWORD *v10; // rbx
  _QWORD *v11; // rax
  _QWORD *v12; // rbx
  _QWORD *v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // rax
  _QWORD *v18; // rbx
  _QWORD *v19; // rax
  _QWORD *v20; // rbx
  _QWORD *v21; // rax
  _QWORD *v22; // rbx
  _QWORD *v23; // rax
  _QWORD *v24; // rbx
  _QWORD *v25; // rax
  _QWORD *v26; // rbx
  _QWORD *v27; // rax
  _QWORD *v28; // rbx
  _QWORD *v29; // rax
  _QWORD *v30; // rbx
  _QWORD *v31; // rax
  _QWORD *v32; // rbx
  _QWORD *v33; // rax
  _QWORD *v34; // rbx
  _QWORD *v35; // rax
  _QWORD *v36; // rbx
  _QWORD *v37; // rax
  _QWORD *v38; // rbx
  _QWORD *v39; // rax
  _QWORD *v40; // rbx
  _QWORD *v41; // rax
  _QWORD *v42; // rbx
  _QWORD *v43; // rax
  _QWORD *v44; // rbx
  _QWORD *v45; // rax
  _QWORD *v46; // rbx
  _QWORD *v47; // rax
  _QWORD *v48; // rbx
  _QWORD *v49; // rax
  _QWORD *v50; // rbx
  _QWORD *v51; // rax
  _QWORD *v52; // rbx
  _QWORD *v53; // rax
  _QWORD *v54; // rbx
  _QWORD *v55; // rax
  _QWORD *v56; // rbx
  _QWORD *v57; // rax
  _QWORD *v58; // rbx
  _QWORD *v59; // rax
  _QWORD *v60; // rbx
  _QWORD *v61; // rax
  _QWORD *v62; // rbx
  unsigned int *v63; // [rsp+0h] [rbp-20h] BYREF
  __int64 *v64[2]; // [rsp+8h] [rbp-18h] BYREF

  v1 = *(_QWORD *)(a1 + 16);
  switch ( *(_BYTE *)a1 )
  {
    case 4:
      v59 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v59 = (_QWORD *)*v59;
      v60 = (_QWORD *)*v59;
      v63 = (unsigned int *)a1;
      result = sub_1621680((__int64)(v60 + 62), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v60 + 128);
        ++*((_DWORD *)v60 + 129);
      }
      break;
    case 5:
      v57 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v57 = (_QWORD *)*v57;
      v58 = (_QWORD *)*v57;
      v63 = (unsigned int *)a1;
      result = sub_15B7120((__int64)(v58 + 66), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v58 + 136);
        ++*((_DWORD *)v58 + 137);
      }
      break;
    case 6:
      v55 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v55 = (_QWORD *)*v55;
      v56 = (_QWORD *)*v55;
      v63 = (unsigned int *)a1;
      result = sub_15B9210((__int64)(v56 + 70), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v56 + 144);
        ++*((_DWORD *)v56 + 145);
      }
      break;
    case 7:
      v53 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v53 = (_QWORD *)*v53;
      v54 = (_QWORD *)*v53;
      v63 = (unsigned int *)a1;
      result = sub_15B92E0((__int64)(v54 + 74), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v54 + 152);
        ++*((_DWORD *)v54 + 153);
      }
      break;
    case 8:
      v51 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v51 = (_QWORD *)*v51;
      v52 = (_QWORD *)*v51;
      v63 = (unsigned int *)a1;
      result = sub_15B7230((__int64)(v52 + 78), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v52 + 160);
        ++*((_DWORD *)v52 + 161);
      }
      break;
    case 9:
      v49 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v49 = (_QWORD *)*v49;
      v50 = (_QWORD *)*v49;
      v63 = (unsigned int *)a1;
      result = sub_15B7360((__int64)(v50 + 82), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v50 + 168);
        ++*((_DWORD *)v50 + 169);
      }
      break;
    case 0xA:
      v47 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v47 = (_QWORD *)*v47;
      v48 = (_QWORD *)*v47;
      v63 = (unsigned int *)a1;
      result = sub_15B77C0((__int64)(v48 + 86), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v48 + 176);
        ++*((_DWORD *)v48 + 177);
      }
      break;
    case 0xB:
      v45 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v45 = (_QWORD *)*v45;
      v46 = (_QWORD *)*v45;
      v63 = (unsigned int *)a1;
      result = sub_15B78B0((__int64)(v46 + 90), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v46 + 184);
        ++*((_DWORD *)v46 + 185);
      }
      break;
    case 0xC:
      v43 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v43 = (_QWORD *)*v43;
      v44 = (_QWORD *)*v43;
      v63 = (unsigned int *)a1;
      result = sub_15B7AF0((__int64)(v44 + 94), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v44 + 192);
        ++*((_DWORD *)v44 + 193);
      }
      break;
    case 0xD:
      v41 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v41 = (_QWORD *)*v41;
      v42 = (_QWORD *)*v41;
      v63 = (unsigned int *)a1;
      result = sub_15B7D90((__int64)(v42 + 98), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v42 + 200);
        ++*((_DWORD *)v42 + 201);
      }
      break;
    case 0xE:
      v39 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v39 = (_QWORD *)*v39;
      v40 = (_QWORD *)*v39;
      v63 = (unsigned int *)a1;
      result = sub_15B80C0((__int64)(v40 + 102), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v40 + 208);
        ++*((_DWORD *)v40 + 209);
      }
      break;
    case 0xF:
      v37 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v37 = (_QWORD *)*v37;
      v38 = (_QWORD *)*v37;
      v63 = (unsigned int *)a1;
      result = sub_15B81B0((__int64)(v38 + 106), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v38 + 216);
        ++*((_DWORD *)v38 + 217);
      }
      break;
    case 0x10:
    case 0x22:
      v61 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v61 = (_QWORD *)*v61;
      v62 = (_QWORD *)*v61;
      v63 = (unsigned int *)a1;
      result = sub_15B7680((__int64)(v62 + 178), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v62 + 360);
        ++*((_DWORD *)v62 + 361);
      }
      break;
    case 0x11:
      v35 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v35 = (_QWORD *)*v35;
      v36 = (_QWORD *)*v35;
      v63 = (unsigned int *)a1;
      result = sub_15B8340((__int64)(v36 + 110), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v36 + 224);
        ++*((_DWORD *)v36 + 225);
      }
      break;
    case 0x12:
      v33 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v33 = (_QWORD *)*v33;
      v34 = (_QWORD *)*v33;
      v63 = (unsigned int *)a1;
      result = sub_15B86E0((__int64)(v34 + 114), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v34 + 232);
        ++*((_DWORD *)v34 + 233);
      }
      break;
    case 0x13:
      v31 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v31 = (_QWORD *)*v31;
      v32 = (_QWORD *)*v31;
      v63 = (unsigned int *)a1;
      result = sub_15B87F0((__int64)(v32 + 118), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v32 + 240);
        ++*((_DWORD *)v32 + 241);
      }
      break;
    case 0x14:
      v29 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v29 = (_QWORD *)*v29;
      v30 = (_QWORD *)*v29;
      v63 = (unsigned int *)a1;
      result = sub_15B88F0((__int64)(v30 + 122), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v30 + 248);
        ++*((_DWORD *)v30 + 249);
      }
      break;
    case 0x15:
      v27 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v27 = (_QWORD *)*v27;
      v28 = (_QWORD *)*v27;
      v63 = (unsigned int *)a1;
      result = sub_15B8B10((__int64)(v28 + 126), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v28 + 256);
        ++*((_DWORD *)v28 + 257);
      }
      break;
    case 0x16:
      v25 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v25 = (_QWORD *)*v25;
      v26 = (_QWORD *)*v25;
      v63 = (unsigned int *)a1;
      result = sub_15B8C40((__int64)(v26 + 130), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v26 + 264);
        ++*((_DWORD *)v26 + 265);
      }
      break;
    case 0x17:
      v23 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v23 = (_QWORD *)*v23;
      v24 = (_QWORD *)*v23;
      v63 = (unsigned int *)a1;
      result = sub_15B8D30((__int64)(v24 + 134), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v24 + 272);
        ++*((_DWORD *)v24 + 273);
      }
      break;
    case 0x18:
      v21 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v21 = (_QWORD *)*v21;
      v22 = (_QWORD *)*v21;
      v63 = (unsigned int *)a1;
      result = sub_15B8E40((__int64)(v22 + 138), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v22 + 280);
        ++*((_DWORD *)v22 + 281);
      }
      break;
    case 0x19:
      v19 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v19 = (_QWORD *)*v19;
      v20 = (_QWORD *)*v19;
      v63 = (unsigned int *)a1;
      result = sub_15B8FB0((__int64)(v20 + 142), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v20 + 288);
        ++*((_DWORD *)v20 + 289);
      }
      break;
    case 0x1A:
      v17 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v17 = (_QWORD *)*v17;
      v18 = (_QWORD *)*v17;
      v63 = (unsigned int *)a1;
      result = sub_15B9100((__int64)(v18 + 146), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v18 + 296);
        ++*((_DWORD *)v18 + 297);
      }
      break;
    case 0x1B:
      v15 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v15 = (_QWORD *)*v15;
      v16 = (_QWORD *)*v15;
      v63 = (unsigned int *)a1;
      result = sub_15B93D0((__int64)(v16 + 150), &v63, (unsigned int ***)v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v16 + 304);
        ++*((_DWORD *)v16 + 305);
      }
      break;
    case 0x1C:
      v13 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v13 = (_QWORD *)*v13;
      v14 = (_QWORD *)*v13;
      v63 = (unsigned int *)a1;
      result = sub_15B9520((__int64)(v14 + 154), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v14 + 312);
        ++*((_DWORD *)v14 + 313);
      }
      break;
    case 0x1D:
      v11 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v11 = (_QWORD *)*v11;
      v12 = (_QWORD *)*v11;
      v63 = (unsigned int *)a1;
      result = sub_15B9650((__int64)(v12 + 158), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v12 + 320);
        ++*((_DWORD *)v12 + 321);
      }
      break;
    case 0x1E:
      v9 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v9 = (_QWORD *)*v9;
      v10 = (_QWORD *)*v9;
      v63 = (unsigned int *)a1;
      result = sub_15B9760((__int64)(v10 + 162), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v10 + 328);
        ++*((_DWORD *)v10 + 329);
      }
      break;
    case 0x1F:
      v7 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v7 = (_QWORD *)*v7;
      v8 = (_QWORD *)*v7;
      v63 = (unsigned int *)a1;
      result = sub_15B89F0((__int64)(v8 + 166), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v8 + 336);
        ++*((_DWORD *)v8 + 337);
      }
      break;
    case 0x20:
      v5 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v5 = (_QWORD *)*v5;
      v6 = (_QWORD *)*v5;
      v63 = (unsigned int *)a1;
      result = sub_15B79C0((__int64)(v6 + 170), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v6 + 344);
        ++*((_DWORD *)v6 + 345);
      }
      break;
    case 0x21:
      v2 = (_QWORD *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v1 & 4) != 0 )
        v2 = (_QWORD *)*v2;
      v3 = (_QWORD *)*v2;
      v63 = (unsigned int *)a1;
      result = sub_15B7F60((__int64)(v3 + 174), (__int64 *)&v63, v64);
      if ( (_BYTE)result )
      {
        *v64[0] = -16;
        --*((_DWORD *)v3 + 352);
        ++*((_DWORD *)v3 + 353);
      }
      break;
  }
  return result;
}
