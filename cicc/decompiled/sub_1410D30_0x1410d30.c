// Function: sub_1410D30
// Address: 0x1410d30
//
_QWORD *__fastcall sub_1410D30(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // rdx
  __int64 v9; // r10
  __int64 v11; // r11
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdi
  unsigned __int64 *v27; // r15
  __int64 v28; // rax
  unsigned __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rax
  _QWORD *v33; // r15
  __int64 v34; // rdx
  unsigned __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rdi
  unsigned __int64 *v44; // r14
  __int64 v45; // rax
  unsigned __int64 v46; // rcx
  __int64 v47; // rsi
  __int64 v48; // rsi
  __int64 v49; // [rsp+8h] [rbp-98h]
  __int64 v50; // [rsp+10h] [rbp-90h]
  __int64 v51; // [rsp+18h] [rbp-88h]
  __int64 v52; // [rsp+18h] [rbp-88h]
  __int64 v53; // [rsp+18h] [rbp-88h]
  __int64 v54; // [rsp+20h] [rbp-80h]
  __int64 v55; // [rsp+20h] [rbp-80h]
  __int64 v56; // [rsp+20h] [rbp-80h]
  __int64 v57; // [rsp+28h] [rbp-78h]
  _BYTE v58[16]; // [rsp+30h] [rbp-70h] BYREF
  __int16 v59; // [rsp+40h] [rbp-60h]
  _QWORD v60[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v61; // [rsp+60h] [rbp-40h]

  v4 = sub_1410110((__int64)a1, *(_QWORD **)(a2 - 48));
  v6 = v5;
  v7 = v4;
  v9 = sub_1410110((__int64)a1, *(_QWORD **)(a2 - 24));
  v57 = v8;
  if ( v9 == 0 || v7 == 0 || v6 == 0 || !v8 )
    return 0;
  if ( v8 == v6 && v7 == v9 )
    return (_QWORD *)v7;
  v11 = *(_QWORD *)(a2 - 72);
  v59 = 257;
  if ( *(_BYTE *)(v11 + 16) > 0x10u || *(_BYTE *)(v7 + 16) > 0x10u || *(_BYTE *)(v9 + 16) > 0x10u )
  {
    v51 = v9;
    v54 = v11;
    v61 = 257;
    v16 = sub_1648A60(56, 3);
    v12 = (_QWORD *)v16;
    if ( v16 )
    {
      v49 = v51;
      v50 = v54;
      v52 = v16 - 72;
      v55 = v16;
      sub_15F1EA0(v16, *(_QWORD *)v7, 55, v16 - 72, 3, 0);
      if ( *(v12 - 9) )
      {
        v17 = *(v12 - 8);
        v18 = *(v12 - 7) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v18 = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = *(_QWORD *)(v17 + 16) & 3LL | v18;
      }
      *(v12 - 9) = v50;
      v19 = *(_QWORD *)(v50 + 8);
      *(v12 - 8) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = (unsigned __int64)(v12 - 8) | *(_QWORD *)(v19 + 16) & 3LL;
      *(v12 - 7) = (v50 + 8) | *(v12 - 7) & 3LL;
      *(_QWORD *)(v50 + 8) = v52;
      if ( *(v12 - 6) )
      {
        v20 = *(v12 - 5);
        v21 = *(v12 - 4) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v21 = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = *(_QWORD *)(v20 + 16) & 3LL | v21;
      }
      *(v12 - 6) = v7;
      v22 = *(_QWORD *)(v7 + 8);
      *(v12 - 5) = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = (unsigned __int64)(v12 - 5) | *(_QWORD *)(v22 + 16) & 3LL;
      *(v12 - 4) = *(v12 - 4) & 3LL | (v7 + 8);
      *(_QWORD *)(v7 + 8) = v12 - 6;
      if ( *(v12 - 3) )
      {
        v23 = *(v12 - 2);
        v24 = *(v12 - 1) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v24 = v23;
        if ( v23 )
          *(_QWORD *)(v23 + 16) = *(_QWORD *)(v23 + 16) & 3LL | v24;
      }
      *(v12 - 3) = v49;
      v25 = *(_QWORD *)(v49 + 8);
      *(v12 - 2) = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = (unsigned __int64)(v12 - 2) | *(_QWORD *)(v25 + 16) & 3LL;
      *(v12 - 1) = *(v12 - 1) & 3LL | (v49 + 8);
      *(_QWORD *)(v49 + 8) = v12 - 3;
      sub_164B780(v12, v60);
    }
    else
    {
      v55 = 0;
    }
    v26 = a1[4];
    if ( v26 )
    {
      v27 = (unsigned __int64 *)a1[5];
      sub_157E9D0(v26 + 40, v12);
      v28 = v12[3];
      v29 = *v27;
      v12[4] = v27;
      v29 &= 0xFFFFFFFFFFFFFFF8LL;
      v12[3] = v29 | v28 & 7;
      *(_QWORD *)(v29 + 8) = v12 + 3;
      *v27 = *v27 & 7 | (unsigned __int64)(v12 + 3);
    }
    sub_164B780(v55, v58);
    v30 = a1[3];
    if ( v30 )
    {
      v60[0] = a1[3];
      sub_1623A60(v60, v30, 2);
      if ( v12[6] )
        sub_161E7C0(v12 + 6);
      v31 = v60[0];
      v12[6] = v60[0];
      if ( v31 )
        sub_1623210(v60, v31, v12 + 6);
    }
  }
  else
  {
    v12 = (_QWORD *)sub_15A2DC0(v11, v7, v9, 0);
    v13 = sub_14DBA30(v12, a1[11], 0);
    if ( v13 )
      v12 = (_QWORD *)v13;
  }
  v14 = *(_QWORD *)(a2 - 72);
  v59 = 257;
  if ( *(_BYTE *)(v14 + 16) > 0x10u || *(_BYTE *)(v6 + 16) > 0x10u || *(_BYTE *)(v57 + 16) > 0x10u )
  {
    v61 = 257;
    v32 = sub_1648A60(56, 3);
    v33 = (_QWORD *)v32;
    if ( v32 )
    {
      v53 = v32 - 72;
      v56 = v32;
      sub_15F1EA0(v32, *(_QWORD *)v6, 55, v32 - 72, 3, 0);
      if ( *(v33 - 9) )
      {
        v34 = *(v33 - 8);
        v35 = *(v33 - 7) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v35 = v34;
        if ( v34 )
          *(_QWORD *)(v34 + 16) = *(_QWORD *)(v34 + 16) & 3LL | v35;
      }
      *(v33 - 9) = v14;
      v36 = *(_QWORD *)(v14 + 8);
      *(v33 - 8) = v36;
      if ( v36 )
        *(_QWORD *)(v36 + 16) = (unsigned __int64)(v33 - 8) | *(_QWORD *)(v36 + 16) & 3LL;
      *(v33 - 7) = (v14 + 8) | *(v33 - 7) & 3LL;
      *(_QWORD *)(v14 + 8) = v53;
      if ( *(v33 - 6) )
      {
        v37 = *(v33 - 5);
        v38 = *(v33 - 4) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v38 = v37;
        if ( v37 )
          *(_QWORD *)(v37 + 16) = *(_QWORD *)(v37 + 16) & 3LL | v38;
      }
      *(v33 - 6) = v6;
      v39 = *(_QWORD *)(v6 + 8);
      *(v33 - 5) = v39;
      if ( v39 )
        *(_QWORD *)(v39 + 16) = (unsigned __int64)(v33 - 5) | *(_QWORD *)(v39 + 16) & 3LL;
      *(v33 - 4) = *(v33 - 4) & 3LL | (v6 + 8);
      *(_QWORD *)(v6 + 8) = v33 - 6;
      if ( *(v33 - 3) )
      {
        v40 = *(v33 - 2);
        v41 = *(v33 - 1) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v41 = v40;
        if ( v40 )
          *(_QWORD *)(v40 + 16) = *(_QWORD *)(v40 + 16) & 3LL | v41;
      }
      *(v33 - 3) = v57;
      v42 = *(_QWORD *)(v57 + 8);
      *(v33 - 2) = v42;
      if ( v42 )
        *(_QWORD *)(v42 + 16) = (unsigned __int64)(v33 - 2) | *(_QWORD *)(v42 + 16) & 3LL;
      *(v33 - 1) = *(v33 - 1) & 3LL | (v57 + 8);
      *(_QWORD *)(v57 + 8) = v33 - 3;
      sub_164B780(v33, v60);
    }
    else
    {
      v56 = 0;
    }
    v43 = a1[4];
    if ( v43 )
    {
      v44 = (unsigned __int64 *)a1[5];
      sub_157E9D0(v43 + 40, v33);
      v45 = v33[3];
      v46 = *v44;
      v33[4] = v44;
      v46 &= 0xFFFFFFFFFFFFFFF8LL;
      v33[3] = v46 | v45 & 7;
      *(_QWORD *)(v46 + 8) = v33 + 3;
      *v44 = *v44 & 7 | (unsigned __int64)(v33 + 3);
    }
    sub_164B780(v56, v58);
    v47 = a1[3];
    if ( v47 )
    {
      v60[0] = a1[3];
      sub_1623A60(v60, v47, 2);
      if ( v33[6] )
        sub_161E7C0(v33 + 6);
      v48 = v60[0];
      v33[6] = v60[0];
      if ( v48 )
        sub_1623210(v60, v48, v33 + 6);
    }
  }
  else
  {
    v15 = sub_15A2DC0(v14, v6, v57, 0);
    sub_14DBA30(v15, a1[11], 0);
  }
  return v12;
}
