// Function: sub_247C1B0
// Address: 0x247c1b0
//
void __fastcall sub_247C1B0(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r10d
  __int64 v3; // rax
  __int64 v4; // rbx
  _BYTE *v5; // r11
  unsigned int v6; // r15d
  unsigned int v7; // ebx
  __int64 v8; // r9
  __int64 v9; // r8
  const char *v10; // rax
  char *v11; // rax
  _BYTE *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r9
  const char *v15; // r11
  unsigned int v16; // r10d
  __int64 v17; // r8
  unsigned int v18; // r15d
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  unsigned int *v21; // rax
  _BYTE *v22; // rax
  __int64 v23; // r15
  __int64 *v24; // rdx
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rax
  unsigned __int64 v28; // r10
  _BYTE *v29; // r11
  _BYTE *v30; // rax
  __int64 v31; // rax
  bool v32; // al
  _QWORD *v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  const char *v37; // rcx
  unsigned __int64 v38; // rcx
  __int64 v39; // r15
  const char *v40; // [rsp+8h] [rbp-148h]
  __int64 v41; // [rsp+10h] [rbp-140h]
  const char *v42; // [rsp+10h] [rbp-140h]
  unsigned int v43; // [rsp+10h] [rbp-140h]
  __int64 v44; // [rsp+18h] [rbp-138h]
  __int64 v45; // [rsp+18h] [rbp-138h]
  unsigned int v46; // [rsp+18h] [rbp-138h]
  const char *v47; // [rsp+18h] [rbp-138h]
  __int64 v48; // [rsp+18h] [rbp-138h]
  unsigned int v49; // [rsp+20h] [rbp-130h]
  __int64 v50; // [rsp+20h] [rbp-130h]
  _BYTE *v51; // [rsp+20h] [rbp-130h]
  unsigned __int64 v52; // [rsp+20h] [rbp-130h]
  _BYTE *v53; // [rsp+20h] [rbp-130h]
  unsigned __int64 v54; // [rsp+20h] [rbp-130h]
  __int64 v55; // [rsp+28h] [rbp-128h]
  __int64 v56; // [rsp+28h] [rbp-128h]
  _BYTE *v57; // [rsp+28h] [rbp-128h]
  unsigned __int64 v58; // [rsp+28h] [rbp-128h]
  _BYTE v59[32]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v60; // [rsp+50h] [rbp-100h]
  const char *v61; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v62; // [rsp+68h] [rbp-E8h]
  _BYTE v63[16]; // [rsp+70h] [rbp-E0h] BYREF
  __int16 v64; // [rsp+80h] [rbp-D0h]
  unsigned int *v65[24]; // [rsp+90h] [rbp-C0h] BYREF

  sub_23D0AB0((__int64)v65, a2, 0, 0, 0);
  v2 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL) + 32LL);
  v3 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_DWORD *)(v3 + 32) <= 0x40u )
    v4 = *(_QWORD *)(v3 + 24);
  else
    v4 = **(_QWORD **)(v3 + 24);
  v5 = v63;
  v6 = v4;
  v7 = v4 & 1;
  v8 = 257;
  v62 = 0x800000000LL;
  v9 = 0;
  v10 = v63;
  v60 = 257;
  v61 = v63;
  if ( v2 > v7 )
  {
    while ( 1 )
    {
      v11 = (char *)&v10[4 * v9];
      *(_DWORD *)v11 = v7;
      *((_DWORD *)v11 + 1) = v7;
      v7 += 2;
      v9 = (unsigned int)(v62 + 2);
      LODWORD(v62) = v62 + 2;
      if ( v2 <= v7 )
        break;
      if ( v9 + 2 > (unsigned __int64)HIDWORD(v62) )
      {
        v46 = v2;
        v53 = v5;
        sub_C8D5F0((__int64)&v61, v5, v9 + 2, 4u, v9, v8);
        v9 = (unsigned int)v62;
        v2 = v46;
        v5 = v53;
      }
      v10 = v61;
    }
    v37 = v61;
  }
  else
  {
    v9 = 0;
    v37 = v63;
  }
  v40 = v5;
  v49 = v2;
  v41 = (__int64)v37;
  v44 = v9;
  v55 = sub_24723A0((__int64)a1, a2, 0);
  v12 = (_BYTE *)sub_ACADE0(*(__int64 ***)(v55 + 8));
  v13 = sub_A83CB0(v65, (_BYTE *)v55, v12, v41, v44, (__int64)v59);
  v15 = v40;
  v56 = v13;
  v16 = v49;
  if ( v61 != v40 )
  {
    _libc_free((unsigned __int64)v61);
    v15 = v40;
    v16 = v49;
  }
  v17 = 257;
  v61 = v15;
  v18 = (v6 >> 4) & 1;
  v62 = 0x800000000LL;
  v19 = (unsigned __int64)v15;
  v20 = 0;
  v60 = 257;
  if ( v16 > v18 )
  {
    while ( 1 )
    {
      v21 = (unsigned int *)(v19 + 4 * v20);
      *v21 = v18;
      v21[1] = v18;
      v18 += 2;
      v20 = (unsigned int)(v62 + 2);
      LODWORD(v62) = v62 + 2;
      if ( v16 <= v18 )
        break;
      if ( v20 + 2 > (unsigned __int64)HIDWORD(v62) )
      {
        v43 = v16;
        v47 = v15;
        sub_C8D5F0((__int64)&v61, v15, v20 + 2, 4u, v17, v14);
        v20 = (unsigned int)v62;
        v16 = v43;
        v15 = v47;
      }
      v19 = (unsigned __int64)v61;
    }
    v38 = (unsigned __int64)v61;
    v39 = (unsigned int)v20;
  }
  else
  {
    v38 = (unsigned __int64)v15;
    v39 = 0;
  }
  v42 = v15;
  v45 = v38;
  v50 = sub_24723A0((__int64)a1, a2, 1u);
  v22 = (_BYTE *)sub_ACADE0(*(__int64 ***)(v50 + 8));
  v23 = sub_A83CB0(v65, (_BYTE *)v50, v22, v45, v39, (__int64)v59);
  if ( v61 != v42 )
    _libc_free((unsigned __int64)v61);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v24 = *(__int64 **)(a2 - 8);
  else
    v24 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v25 = sub_246EE10((__int64)a1, *v24);
  if ( !*(_DWORD *)(a1[1] + 4) )
    v25 = 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v26 = *(_QWORD *)(a2 - 8);
  else
    v26 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v27 = sub_246EE10((__int64)a1, *(_QWORD *)(v26 + 32));
  v28 = v23;
  v29 = (_BYTE *)v27;
  if ( v56 )
  {
    v51 = (_BYTE *)v27;
    v30 = (_BYTE *)sub_2464970(a1, v65, v23, *(_QWORD *)(v56 + 8), 0);
    v61 = "_msprop";
    v23 = (__int64)v30;
    v64 = 259;
    v31 = sub_A82480(v65, (_BYTE *)v56, v30, (__int64)&v61);
    v29 = v51;
    v28 = v31;
  }
  if ( *(_DWORD *)(a1[1] + 4) )
  {
    if ( v25 )
    {
      if ( *v29 > 0x15u || (v52 = v28, v57 = v29, v32 = sub_AC30F0((__int64)v29), v29 = v57, v28 = v52, !v32) )
      {
        v64 = 257;
        v54 = v28;
        v48 = (__int64)v29;
        v35 = sub_2465600((__int64)a1, v23, (__int64)v65, (__int64)&v61);
        v64 = 257;
        v36 = sub_B36550(v65, v35, v48, v25, (__int64)&v61, 0);
        v28 = v54;
        v25 = v36;
      }
    }
    else
    {
      v25 = (__int64)v29;
    }
  }
  v58 = v28;
  v33 = sub_2463540(a1, *(_QWORD *)(a2 + 8));
  v34 = sub_2464970(a1, v65, v58, (__int64)v33, 0);
  sub_246EF60((__int64)a1, a2, v34);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_246F1C0((__int64)a1, a2, v25);
  sub_F94A20(v65, a2);
}
