// Function: sub_17D69B0
// Address: 0x17d69b0
//
_QWORD *__fastcall sub_17D69B0(_QWORD *a1, __int64 *a2, __int64 a3)
{
  unsigned __int64 v5; // r12
  __int64 **v6; // rbx
  __int64 v7; // rax
  unsigned int v8; // r8d
  __int64 v9; // rdx
  __int64 *v10; // r15
  __int64 v11; // rsi
  char v12; // al
  unsigned __int64 v13; // r12
  unsigned int v14; // r8d
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 *v23; // rax
  __int64 v24; // rax
  __int128 v25; // rdi
  __int64 *v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v32; // rax
  unsigned int v33; // eax
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rsi
  __int64 v41; // rdx
  unsigned __int8 *v42; // rsi
  __int64 v43; // [rsp+18h] [rbp-C8h]
  unsigned int v44; // [rsp+24h] [rbp-BCh]
  __int64 v45; // [rsp+28h] [rbp-B8h]
  __int64 *v46; // [rsp+28h] [rbp-B8h]
  __int64 v47; // [rsp+30h] [rbp-B0h]
  int v48; // [rsp+30h] [rbp-B0h]
  unsigned int v49; // [rsp+40h] [rbp-A0h]
  int v50; // [rsp+44h] [rbp-9Ch]
  unsigned __int64 v52; // [rsp+50h] [rbp-90h]
  int v53; // [rsp+58h] [rbp-88h]
  __int64 v54; // [rsp+58h] [rbp-88h]
  int v55; // [rsp+58h] [rbp-88h]
  __int64 v56; // [rsp+58h] [rbp-88h]
  unsigned int v57; // [rsp+58h] [rbp-88h]
  __int64 v58; // [rsp+68h] [rbp-78h] BYREF
  __int64 v59; // [rsp+70h] [rbp-70h] BYREF
  __int16 v60; // [rsp+80h] [rbp-60h]
  _QWORD v61[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v62; // [rsp+A0h] [rbp-40h]

  v43 = sub_1632FA0(*(_QWORD *)(a1[1] + 40LL));
  v5 = (*a2 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  v52 = sub_1389B50(a2);
  if ( v52 == v5 )
  {
    v28 = 0;
    goto LABEL_17;
  }
  v6 = (__int64 **)v5;
  v50 = 192;
  v44 = 64;
  v49 = 0;
  do
  {
    while ( 1 )
    {
      v10 = *v6;
      v11 = **v6;
      v54 = *a2;
      v12 = *(_BYTE *)(v11 + 8);
      v13 = 0xAAAAAAAAAAAAAAABLL
          * ((__int64)((__int64)&v6[3 * (*(_DWORD *)((v54 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)]
                     - (v54 & 0xFFFFFFFFFFFFFFF8LL)) >> 3);
      v14 = *(_DWORD *)(*(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 64) + 12LL) - 1;
      if ( v12 == 16 )
      {
        if ( (unsigned __int8)(*(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL) - 1) > 5u )
          goto LABEL_10;
        goto LABEL_4;
      }
      if ( (unsigned __int8)(v12 - 1) > 5u )
        break;
LABEL_4:
      if ( v44 > 0xBF )
        goto LABEL_10;
      v53 = *(_DWORD *)(*(_QWORD *)((v54 & 0xFFFFFFFFFFFFFFF8LL) + 64) + 12LL) - 1;
      v7 = sub_17CE860((__int64)a1, v11, (__int64 *)a3, v44);
      v44 += 16;
      v8 = v53;
      v9 = v7;
LABEL_6:
      if ( v8 <= (unsigned int)v13 )
        goto LABEL_15;
LABEL_7:
      v6 += 3;
      if ( (__int64 **)v52 == v6 )
        goto LABEL_16;
    }
    if ( v12 == 11 )
    {
      v48 = *(_DWORD *)(*(_QWORD *)((v54 & 0xFFFFFFFFFFFFFFF8LL) + 64) + 12LL) - 1;
      v33 = sub_1643030(v11);
      v14 = v48;
      if ( v33 > 0x40 )
        goto LABEL_10;
    }
    else if ( v12 != 15 )
    {
      goto LABEL_10;
    }
    if ( v49 <= 0x3F )
    {
      v57 = v14;
      v32 = sub_17CE860((__int64)a1, v11, (__int64 *)a3, v49);
      v49 += 8;
      v8 = v57;
      v9 = v32;
      goto LABEL_6;
    }
LABEL_10:
    if ( v14 > (unsigned int)v13 )
      goto LABEL_7;
    v55 = sub_12BE0A0(v43, v11);
    v15 = *v10;
    v60 = 257;
    v47 = v15;
    v16 = a1[2];
    v17 = *(_QWORD *)(v16 + 224);
    v18 = *(_QWORD *)(v16 + 176);
    v19 = *(_QWORD *)v17;
    if ( v18 != *(_QWORD *)v17 )
    {
      if ( *(_BYTE *)(v17 + 16) > 0x10u )
      {
        v34 = *(_QWORD *)(v16 + 224);
        v62 = 257;
        v35 = sub_15FDFF0(v34, v18, (__int64)v61, 0);
        v36 = *(_QWORD *)(a3 + 8);
        v17 = v35;
        if ( v36 )
        {
          v46 = *(__int64 **)(a3 + 16);
          sub_157E9D0(v36 + 40, v35);
          v37 = *v46;
          v38 = *(_QWORD *)(v17 + 24) & 7LL;
          *(_QWORD *)(v17 + 32) = v46;
          v37 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v17 + 24) = v37 | v38;
          *(_QWORD *)(v37 + 8) = v17 + 24;
          *v46 = *v46 & 7 | (v17 + 24);
        }
        sub_164B780(v17, &v59);
        v39 = *(_QWORD *)a3;
        if ( *(_QWORD *)a3 )
        {
          v58 = *(_QWORD *)a3;
          sub_1623A60((__int64)&v58, v39, 2);
          v40 = *(_QWORD *)(v17 + 48);
          v41 = v17 + 48;
          if ( v40 )
          {
            sub_161E7C0(v17 + 48, v40);
            v41 = v17 + 48;
          }
          v42 = (unsigned __int8 *)v58;
          *(_QWORD *)(v17 + 48) = v58;
          if ( v42 )
            sub_1623210((__int64)&v58, v42, v41);
        }
        v19 = *(_QWORD *)(a1[2] + 176LL);
      }
      else
      {
        v17 = sub_15A4A70(*(__int64 ****)(v16 + 224), v18);
        v19 = *(_QWORD *)(a1[2] + 176LL);
      }
    }
    v62 = 257;
    v20 = sub_15A0680(v19, v50, 0);
    v21 = sub_12899C0((__int64 *)a3, v17, v20, (__int64)v61, 0, 0);
    v61[0] = "_msarg";
    v22 = (_QWORD *)a1[3];
    v45 = v21;
    v62 = 259;
    v23 = sub_17CD8D0(v22, v47);
    v24 = sub_1646BA0(v23, 0);
    v9 = sub_12AA3B0((__int64 *)a3, 0x2Eu, v45, v24, (__int64)v61);
    v50 += (v55 + 7) & 0xFFFFFFF8;
LABEL_15:
    *(_QWORD *)&v25 = a1[3];
    *((_QWORD *)&v25 + 1) = v10;
    v56 = v9;
    v6 += 3;
    v26 = sub_17D4DA0(v25);
    v27 = sub_12A8F50((__int64 *)a3, (__int64)v26, v56, 0);
    sub_15F9450((__int64)v27, 8u);
  }
  while ( (__int64 **)v52 != v6 );
LABEL_16:
  v28 = (unsigned int)(v50 - 192);
LABEL_17:
  v29 = sub_1643360(*(_QWORD **)(a3 + 24));
  v30 = sub_159C470(v29, v28, 0);
  return sub_12A8F50((__int64 *)a3, v30, *(_QWORD *)(a1[2] + 232LL), 0);
}
