// Function: sub_2412430
// Address: 0x2412430
//
__int64 __fastcall sub_2412430(_QWORD *a1, unsigned __int64 a2, char a3, __int64 a4, __int16 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rsi
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 *v12; // rdi
  __int64 v13; // r10
  __int64 (__fastcall *v14)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rsi
  unsigned __int8 v18; // dl
  unsigned __int64 v19; // rax
  __int64 **v20; // r15
  __int64 (__fastcall *v21)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v22; // r14
  int v24; // r14d
  __int64 v25; // r14
  unsigned int *v26; // r14
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v29; // r14
  int v30; // r13d
  unsigned int *v31; // rbx
  unsigned int *v32; // r13
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // rdi
  unsigned __int8 *v36; // r14
  __int64 (__fastcall *v37)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v38; // r15
  _BYTE *v39; // rax
  __int64 v40; // rax
  unsigned int *v41; // r13
  unsigned int *v42; // r14
  __int64 v43; // rdx
  unsigned int v44; // esi
  __int64 **v46; // [rsp+8h] [rbp-128h]
  unsigned int *v47; // [rsp+8h] [rbp-128h]
  __int64 v48; // [rsp+8h] [rbp-128h]
  _BYTE v49[32]; // [rsp+10h] [rbp-120h] BYREF
  __int16 v50; // [rsp+30h] [rbp-100h]
  _BYTE v51[32]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v52; // [rsp+60h] [rbp-D0h]
  unsigned int *v53; // [rsp+70h] [rbp-C0h] BYREF
  unsigned int v54; // [rsp+78h] [rbp-B8h]
  char v55; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v56; // [rsp+A8h] [rbp-88h]
  __int64 v57; // [rsp+B0h] [rbp-80h]
  __int64 v58; // [rsp+C0h] [rbp-70h]
  __int64 v59; // [rsp+C8h] [rbp-68h]
  __int64 v60; // [rsp+D0h] [rbp-60h]
  int v61; // [rsp+D8h] [rbp-58h]
  void *v62; // [rsp+F0h] [rbp-40h]

  if ( !a4 )
    BUG();
  sub_2412230((__int64)&v53, *(_QWORD *)(a4 + 16), a4, a5, 0, a6, 0, 0);
  v7 = (__int64)sub_240FA00((__int64)a1, a2, (__int64 *)&v53);
  v8 = v7;
  v9 = *(_QWORD *)(a1[115] + 16LL);
  if ( v9 )
  {
    v10 = a1[8];
    v52 = 257;
    v11 = (_BYTE *)sub_ACD640(v10, v9, 0);
    v8 = sub_929C50(&v53, (_BYTE *)v7, v11, (__int64)v51, 0, 0);
  }
  v12 = (__int64 *)a1[1];
  v50 = 257;
  v13 = sub_BCE3C0(v12, 0);
  if ( v13 == *(_QWORD *)(v8 + 8) )
  {
    v16 = v8;
    goto LABEL_11;
  }
  v14 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v58 + 120LL);
  if ( v14 != sub_920130 )
  {
    v48 = v13;
    v40 = v14(v58, 48u, (_BYTE *)v8, v13);
    v13 = v48;
    v16 = v40;
    goto LABEL_10;
  }
  if ( *(_BYTE *)v8 <= 0x15u )
  {
    v46 = (__int64 **)v13;
    if ( (unsigned __int8)sub_AC4810(0x30u) )
      v15 = sub_ADAB70(48, v8, v46, 0);
    else
      v15 = sub_AA93C0(0x30u, v8, (__int64)v46);
    v13 = (__int64)v46;
    v16 = v15;
LABEL_10:
    if ( v16 )
      goto LABEL_11;
  }
  v52 = 257;
  v16 = sub_B51D30(48, v8, v13, (__int64)v51, 0, 0);
  if ( (unsigned __int8)sub_920620(v16) )
  {
    v24 = v61;
    if ( v60 )
      sub_B99FD0(v16, 3u, v60);
    sub_B45150(v16, v24);
  }
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v59 + 16LL))(v59, v16, v49, v56, v57);
  v25 = 4LL * v54;
  v47 = &v53[v25];
  if ( v53 != &v53[v25] )
  {
    v26 = v53;
    do
    {
      v27 = *((_QWORD *)v26 + 1);
      v28 = *v26;
      v26 += 4;
      sub_B99FD0(v16, v28, v27);
    }
    while ( v47 != v26 );
  }
LABEL_11:
  if ( !(unsigned __int8)sub_240D530() )
    goto LABEL_23;
  v17 = *(_QWORD *)(a1[115] + 24LL);
  if ( v17 )
  {
    v52 = 257;
    v39 = (_BYTE *)sub_ACD640(a1[8], v17, 0);
    v7 = sub_929C50(&v53, (_BYTE *)v7, v39, (__int64)v51, 0, 0);
  }
  v18 = 0;
  if ( 1LL << a3 )
  {
    _BitScanReverse64(&v19, 1LL << a3);
    v18 = 63 - (v19 ^ 0x3F);
  }
  if ( v18 < (unsigned __int8)byte_4FE3AA8 )
  {
    v35 = a1[8];
    v50 = 257;
    v36 = (unsigned __int8 *)sub_ACD640(v35, -1LL << byte_4FE3AA8, 0);
    v37 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v58 + 16LL);
    if ( v37 == sub_9202E0 )
    {
      if ( *(_BYTE *)v7 > 0x15u || *v36 > 0x15u )
        goto LABEL_54;
      if ( (unsigned __int8)sub_AC47B0(28) )
        v38 = sub_AD5570(28, v7, v36, 0, 0);
      else
        v38 = sub_AABE40(0x1Cu, (unsigned __int8 *)v7, v36);
    }
    else
    {
      v38 = v37(v58, 28u, (_BYTE *)v7, v36);
    }
    if ( v38 )
    {
LABEL_50:
      v7 = v38;
      goto LABEL_17;
    }
LABEL_54:
    v52 = 257;
    v38 = sub_B504D0(28, v7, (__int64)v36, (__int64)v51, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v59 + 16LL))(
      v59,
      v38,
      v49,
      v56,
      v57);
    v41 = v53;
    v42 = &v53[4 * v54];
    if ( v53 != v42 )
    {
      do
      {
        v43 = *((_QWORD *)v41 + 1);
        v44 = *v41;
        v41 += 4;
        sub_B99FD0(v38, v44, v43);
      }
      while ( v42 != v41 );
    }
    goto LABEL_50;
  }
LABEL_17:
  v20 = (__int64 **)a1[4];
  v50 = 257;
  if ( v20 == *(__int64 ***)(v7 + 8) )
    goto LABEL_23;
  v21 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v58 + 120LL);
  if ( v21 != sub_920130 )
  {
    v22 = v21(v58, 48u, (_BYTE *)v7, (__int64)v20);
    goto LABEL_22;
  }
  if ( *(_BYTE *)v7 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x30u) )
      v22 = sub_ADAB70(48, v7, v20, 0);
    else
      v22 = sub_AA93C0(0x30u, v7, (__int64)v20);
LABEL_22:
    if ( v22 )
      goto LABEL_23;
  }
  v52 = 257;
  v29 = sub_B51D30(48, v7, (__int64)v20, (__int64)v51, 0, 0);
  if ( (unsigned __int8)sub_920620(v29) )
  {
    v30 = v61;
    if ( v60 )
      sub_B99FD0(v29, 3u, v60);
    sub_B45150(v29, v30);
  }
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v59 + 16LL))(v59, v29, v49, v56, v57);
  v31 = v53;
  v32 = &v53[4 * v54];
  if ( v53 != v32 )
  {
    do
    {
      v33 = *((_QWORD *)v31 + 1);
      v34 = *v31;
      v31 += 4;
      sub_B99FD0(v29, v34, v33);
    }
    while ( v32 != v31 );
  }
LABEL_23:
  nullsub_61();
  v62 = &unk_49DA100;
  nullsub_63();
  if ( v53 != (unsigned int *)&v55 )
    _libc_free((unsigned __int64)v53);
  return v16;
}
