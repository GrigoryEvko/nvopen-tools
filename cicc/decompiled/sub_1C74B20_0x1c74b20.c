// Function: sub_1C74B20
// Address: 0x1c74b20
//
_QWORD *__fastcall sub_1C74B20(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v7; // rbx
  unsigned __int8 *v8; // rsi
  int v9; // ebx
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned __int8 *v12; // rsi
  _QWORD *v13; // r12
  _QWORD *v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 *v26; // r13
  __int64 v27; // rax
  unsigned __int64 v28; // rcx
  __int64 v29; // rsi
  unsigned __int8 *v30; // rsi
  _QWORD *v31; // rax
  _QWORD **v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 v38; // rdx
  unsigned __int8 *v39; // rsi
  _QWORD *v40; // [rsp-E8h] [rbp-E8h]
  __int64 *v41; // [rsp-E8h] [rbp-E8h]
  _QWORD *v42; // [rsp-E0h] [rbp-E0h]
  __int64 v43; // [rsp-E0h] [rbp-E0h]
  unsigned __int8 *v44; // [rsp-D0h] [rbp-D0h] BYREF
  __int64 v45[2]; // [rsp-C8h] [rbp-C8h] BYREF
  char v46; // [rsp-B8h] [rbp-B8h]
  char v47; // [rsp-B7h] [rbp-B7h]
  unsigned __int8 *v48[2]; // [rsp-A8h] [rbp-A8h] BYREF
  __int16 v49; // [rsp-98h] [rbp-98h]
  unsigned __int8 *v50; // [rsp-88h] [rbp-88h] BYREF
  __int64 v51; // [rsp-80h] [rbp-80h]
  __int64 *v52; // [rsp-78h] [rbp-78h]
  __int64 v53; // [rsp-70h] [rbp-70h]
  __int64 v54; // [rsp-68h] [rbp-68h]
  int v55; // [rsp-60h] [rbp-60h]
  __int64 v56; // [rsp-58h] [rbp-58h]
  __int64 v57; // [rsp-50h] [rbp-50h]

  if ( !a4 )
    BUG();
  v7 = *(_QWORD *)(a4 + 16);
  v51 = v7;
  v50 = 0;
  v53 = sub_157E9C0(v7);
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v52 = (__int64 *)a4;
  if ( a4 != v7 + 40 )
  {
    v8 = *(unsigned __int8 **)(a4 + 24);
    v48[0] = v8;
    if ( v8 )
    {
      sub_1623A60((__int64)v48, (__int64)v8, 2);
      if ( v50 )
        sub_161E7C0((__int64)&v50, (__int64)v50);
      v50 = v48[0];
      if ( v48[0] )
        sub_1623210((__int64)v48, v48[0], (__int64)&v50);
    }
  }
  v47 = 1;
  v45[0] = (__int64)&unk_42D2000;
  v46 = 3;
  v9 = a3 == 0 ? 36 : 40;
  if ( *(_BYTE *)(a1 + 16) > 0x10u || *(_BYTE *)(a2 + 16) > 0x10u )
  {
    v49 = 257;
    v31 = sub_1648A60(56, 2u);
    v10 = (__int64)v31;
    if ( v31 )
    {
      v43 = (__int64)v31;
      v32 = *(_QWORD ***)a1;
      if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 )
      {
        v40 = v32[4];
        v33 = (__int64 *)sub_1643320(*v32);
        v34 = (__int64)sub_16463B0(v33, (unsigned int)v40);
      }
      else
      {
        v34 = sub_1643320(*v32);
      }
      sub_15FEC10(v10, v34, 51, v9, a1, a2, (__int64)v48, 0);
    }
    else
    {
      v43 = 0;
    }
    if ( v51 )
    {
      v41 = v52;
      sub_157E9D0(v51 + 40, v10);
      v35 = *v41;
      v36 = *(_QWORD *)(v10 + 24) & 7LL;
      *(_QWORD *)(v10 + 32) = v41;
      v35 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v10 + 24) = v35 | v36;
      *(_QWORD *)(v35 + 8) = v10 + 24;
      *v41 = *v41 & 7 | (v10 + 24);
    }
    sub_164B780(v43, v45);
    if ( v50 )
    {
      v44 = v50;
      sub_1623A60((__int64)&v44, (__int64)v50, 2);
      v37 = *(_QWORD *)(v10 + 48);
      v38 = v10 + 48;
      if ( v37 )
      {
        sub_161E7C0(v10 + 48, v37);
        v38 = v10 + 48;
      }
      v39 = v44;
      *(_QWORD *)(v10 + 48) = v44;
      if ( v39 )
        sub_1623210((__int64)&v44, v39, v38);
    }
  }
  else
  {
    v10 = sub_15A37B0(v9, (_QWORD *)a1, (_QWORD *)a2, 0);
  }
  v47 = 1;
  v45[0] = (__int64)&unk_42D2000;
  v46 = 3;
  if ( *(_BYTE *)(v10 + 16) <= 0x10u && *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a1 + 16) <= 0x10u )
  {
    v11 = sub_15A2DC0(v10, (__int64 *)a2, a1, 0);
    v12 = v50;
    v13 = (_QWORD *)v11;
LABEL_15:
    if ( v12 )
      sub_161E7C0((__int64)&v50, (__int64)v12);
    return v13;
  }
  v49 = 257;
  v15 = sub_1648A60(56, 3u);
  v13 = v15;
  if ( v15 )
  {
    v42 = v15 - 9;
    v16 = (__int64)v15;
    sub_15F1EA0((__int64)v15, *(_QWORD *)a2, 55, (__int64)(v15 - 9), 3, 0);
    if ( *(v13 - 9) )
    {
      v17 = *(v13 - 8);
      v18 = *(v13 - 7) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v18 = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = *(_QWORD *)(v17 + 16) & 3LL | v18;
    }
    *(v13 - 9) = v10;
    v19 = *(_QWORD *)(v10 + 8);
    *(v13 - 8) = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = (unsigned __int64)(v13 - 8) | *(_QWORD *)(v19 + 16) & 3LL;
    *(v13 - 7) = (v10 + 8) | *(v13 - 7) & 3LL;
    *(_QWORD *)(v10 + 8) = v42;
    if ( *(v13 - 6) )
    {
      v20 = *(v13 - 5);
      v21 = *(v13 - 4) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v21 = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = *(_QWORD *)(v20 + 16) & 3LL | v21;
    }
    *(v13 - 6) = a2;
    v22 = *(_QWORD *)(a2 + 8);
    *(v13 - 5) = v22;
    if ( v22 )
      *(_QWORD *)(v22 + 16) = (unsigned __int64)(v13 - 5) | *(_QWORD *)(v22 + 16) & 3LL;
    *(v13 - 4) = (a2 + 8) | *(v13 - 4) & 3LL;
    *(_QWORD *)(a2 + 8) = v13 - 6;
    if ( *(v13 - 3) )
    {
      v23 = *(v13 - 2);
      v24 = *(v13 - 1) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v24 = v23;
      if ( v23 )
        *(_QWORD *)(v23 + 16) = *(_QWORD *)(v23 + 16) & 3LL | v24;
    }
    *(v13 - 3) = a1;
    v25 = *(_QWORD *)(a1 + 8);
    *(v13 - 2) = v25;
    if ( v25 )
      *(_QWORD *)(v25 + 16) = (unsigned __int64)(v13 - 2) | *(_QWORD *)(v25 + 16) & 3LL;
    *(v13 - 1) = (a1 + 8) | *(v13 - 1) & 3LL;
    *(_QWORD *)(a1 + 8) = v13 - 3;
    sub_164B780((__int64)v13, (__int64 *)v48);
  }
  else
  {
    v16 = 0;
  }
  if ( v51 )
  {
    v26 = (unsigned __int64 *)v52;
    sub_157E9D0(v51 + 40, (__int64)v13);
    v27 = v13[3];
    v28 = *v26;
    v13[4] = v26;
    v28 &= 0xFFFFFFFFFFFFFFF8LL;
    v13[3] = v28 | v27 & 7;
    *(_QWORD *)(v28 + 8) = v13 + 3;
    *v26 = *v26 & 7 | (unsigned __int64)(v13 + 3);
  }
  sub_164B780(v16, v45);
  if ( v50 )
  {
    v48[0] = v50;
    sub_1623A60((__int64)v48, (__int64)v50, 2);
    v29 = v13[6];
    if ( v29 )
      sub_161E7C0((__int64)(v13 + 6), v29);
    v30 = v48[0];
    v13[6] = v48[0];
    if ( v30 )
      sub_1623210((__int64)v48, v30, (__int64)(v13 + 6));
    v12 = v50;
    goto LABEL_15;
  }
  return v13;
}
