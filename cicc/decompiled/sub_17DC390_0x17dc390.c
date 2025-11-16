// Function: sub_17DC390
// Address: 0x17dc390
//
unsigned __int64 __fastcall sub_17DC390(__int128 a1, double a2, double a3, double a4)
{
  unsigned __int128 v4; // kr00_16
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // r15
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r13
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // [rsp+20h] [rbp-130h]
  __int64 *v44; // [rsp+30h] [rbp-120h]
  __int64 v45; // [rsp+30h] [rbp-120h]
  __int64 v46; // [rsp+30h] [rbp-120h]
  __int64 *v47; // [rsp+30h] [rbp-120h]
  __int64 *v48; // [rsp+38h] [rbp-118h]
  __int64 v49; // [rsp+38h] [rbp-118h]
  _QWORD *v50; // [rsp+38h] [rbp-118h]
  const char *v51; // [rsp+40h] [rbp-110h]
  __int64 v52; // [rsp+48h] [rbp-108h]
  __int64 ***v53; // [rsp+48h] [rbp-108h]
  __int64 v54; // [rsp+50h] [rbp-100h] BYREF
  __int16 v55; // [rsp+60h] [rbp-F0h]
  _BYTE v56[16]; // [rsp+70h] [rbp-E0h] BYREF
  __int16 v57; // [rsp+80h] [rbp-D0h]
  __int64 v58[2]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v59; // [rsp+A0h] [rbp-B0h]
  _QWORD v60[2]; // [rsp+B0h] [rbp-A0h] BYREF
  __int16 v61; // [rsp+C0h] [rbp-90h]
  __int64 v62; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v63; // [rsp+D8h] [rbp-78h]
  __int64 *v64; // [rsp+E0h] [rbp-70h]
  _QWORD *v65; // [rsp+E8h] [rbp-68h]

  v4 = a1;
  sub_17CE510((__int64)&v62, *((__int64 *)&a1 + 1), 0, 0, 0);
  v5 = *(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF;
  v6 = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24 * v5);
  v7 = *(_QWORD *)(*((_QWORD *)&a1 + 1) + 24 * (1 - v5));
  if ( *(_DWORD *)(v7 + 32) <= 0x40u )
    v52 = *(_QWORD *)(v7 + 24);
  else
    v52 = **(_QWORD **)(v7 + 24);
  v8 = *(_QWORD *)(*((_QWORD *)&a1 + 1) + 24 * (2 - v5));
  v51 = *(const char **)(*((_QWORD *)&a1 + 1) + 24 * (3 - v5));
  v48 = sub_17CD8D0((_QWORD *)a1, **((_QWORD **)&a1 + 1));
  if ( *(_BYTE *)(a1 + 489) )
  {
    v9 = sub_17CFB40(a1, v6, &v62, v48, v52);
    *((_QWORD *)&a1 + 1) = v51;
    v43 = v10;
    v60[0] = "_msmaskedld";
    v44 = (__int64 *)v9;
    v61 = 259;
    v11 = sub_17D4DA0(a1);
    v12 = sub_15E8010(&v62, v44, v52, v8, (__int64)v11, (__int64)v60);
    *((_QWORD *)&a1 + 1) = *((_QWORD *)&v4 + 1);
    sub_17D4920(a1, *((__int64 **)&v4 + 1), (__int64)v12);
    if ( !byte_4FA4600 )
      goto LABEL_5;
  }
  else
  {
    v34 = sub_17CDAE0((_QWORD *)a1, **((_QWORD **)&a1 + 1));
    sub_17D4920(a1, *((__int64 **)&a1 + 1), v34);
    if ( !byte_4FA4600 )
      goto LABEL_5;
  }
  *((_QWORD *)&a1 + 1) = v6;
  sub_17D5820(a1, *((__int64 *)&v4 + 1));
  *((_QWORD *)&a1 + 1) = v8;
  sub_17D5820(a1, *((__int64 *)&v4 + 1));
LABEL_5:
  v14 = *(_QWORD *)(a1 + 8);
  v15 = *(unsigned int *)(v14 + 156);
  if ( (_DWORD)v15 )
  {
    if ( *(_BYTE *)(a1 + 489) )
    {
      v59 = 257;
      v57 = 257;
      v55 = 257;
      if ( *(_BYTE *)(v8 + 16) > 0x10u )
      {
        v61 = 257;
        v35 = (_QWORD *)sub_15FB530((__int64 *)v8, (__int64)v60, 0, v13);
        v16 = (__int64)sub_17CF870(&v62, v35, &v54);
      }
      else
      {
        v16 = sub_15A2B90((__int64 *)v8, 0, 0, v13, a2, a3, a4);
      }
      *((_QWORD *)&a1 + 1) = v51;
      v17 = sub_12AA3B0(&v62, 0x26u, v16, (__int64)v48, (__int64)v56);
      v18 = sub_17D4DA0(a1);
      v19 = sub_1281C00(&v62, (__int64)v18, v17, (__int64)v58);
      v61 = 257;
      v20 = v19;
      v21 = sub_1643350(v65);
      v22 = sub_159C470(v21, 0, 0);
      v45 = v20;
      v53 = (__int64 ***)sub_156D5F0(&v62, v20, v22, (__int64)v60);
      v23 = *(_QWORD *)(*(_QWORD *)v51 + 32LL);
      if ( (int)v23 > 1 )
      {
        v24 = 1;
        v25 = v45;
        v49 = (unsigned int)(v23 - 2) + 2LL;
        do
        {
          v59 = 257;
          v27 = sub_1643350(v65);
          v28 = sub_159C470(v27, v24, 0);
          if ( *(_BYTE *)(v25 + 16) > 0x10u || *(_BYTE *)(v28 + 16) > 0x10u )
          {
            v46 = v28;
            v61 = 257;
            v29 = sub_1648A60(56, 2u);
            v26 = (__int64)v29;
            if ( v29 )
              sub_15FA320((__int64)v29, (_QWORD *)v25, v46, (__int64)v60, 0);
            if ( v63 )
            {
              v47 = v64;
              sub_157E9D0(v63 + 40, v26);
              v30 = *v47;
              v31 = *(_QWORD *)(v26 + 24) & 7LL;
              *(_QWORD *)(v26 + 32) = v47;
              v30 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v26 + 24) = v30 | v31;
              *(_QWORD *)(v30 + 8) = v26 + 24;
              *v47 = *v47 & 7 | (v26 + 24);
            }
            sub_164B780(v26, v58);
            sub_12A86E0(&v62, v26);
          }
          else
          {
            v26 = sub_15A37D0((_BYTE *)v25, v28, 0);
          }
          v61 = 257;
          ++v24;
          v53 = (__int64 ***)sub_156D390(&v62, (__int64)v53, v26, (__int64)v60);
        }
        while ( v24 != v49 );
        v4 = __PAIR128__(*((unsigned __int64 *)&v4 + 1), a1);
      }
      v59 = 257;
      v61 = 257;
      v50 = sub_156E5B0(&v62, v43, (__int64)v58);
      v38 = sub_17D4880(v4, v51, v36, v37);
      v57 = 257;
      v40 = sub_15A06D0(*v53, (__int64)v51, v39, 257);
      v41 = sub_12AA0C0(&v62, 0x21u, v53, v40, (__int64)v56);
      v42 = sub_156B790(&v62, v41, v38, (__int64)v50, (__int64)v60, 0);
      sub_17D4B80(v4, *((__int64 *)&v4 + 1), v42);
    }
    else
    {
      v32 = sub_15A06D0(*(__int64 ***)(v14 + 184), *((__int64 *)&a1 + 1), v15, v13);
      sub_17D4B80(a1, *((__int64 *)&v4 + 1), v32);
    }
  }
  return sub_17CD270(&v62);
}
