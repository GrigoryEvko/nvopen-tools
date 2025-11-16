// Function: sub_24793E0
// Address: 0x24793e0
//
void __fastcall sub_24793E0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // r11
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r15
  _BYTE *v7; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned __int64 v10; // rbx
  _BYTE *v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // r11
  _BYTE *v14; // r10
  __int64 **v15; // r15
  unsigned int v16; // r13d
  unsigned int v17; // eax
  unsigned __int64 v18; // rax
  __int64 **v19; // r15
  unsigned int v20; // r13d
  unsigned int v21; // eax
  unsigned __int64 v22; // rax
  _BYTE *v23; // r13
  _BYTE *v24; // rdx
  __int64 v25; // rax
  unsigned __int8 *v26; // r15
  __int64 v27; // rbx
  _QWORD *v28; // r12
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int8 *v31; // r13
  __int64 (__fastcall *v32)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  unsigned int *v33; // r13
  unsigned int *v34; // rbx
  __int64 v35; // rdx
  unsigned int v36; // esi
  _BYTE *v37; // [rsp+8h] [rbp-168h]
  __int64 v38; // [rsp+18h] [rbp-158h]
  unsigned __int64 v39; // [rsp+18h] [rbp-158h]
  _BYTE *v40; // [rsp+18h] [rbp-158h]
  __int64 v42; // [rsp+28h] [rbp-148h]
  _BYTE *v43; // [rsp+28h] [rbp-148h]
  _QWORD v44[2]; // [rsp+30h] [rbp-140h] BYREF
  _QWORD v45[2]; // [rsp+40h] [rbp-130h] BYREF
  int v46[8]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v47; // [rsp+70h] [rbp-100h]
  _BYTE v48[32]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v49; // [rsp+A0h] [rbp-D0h]
  unsigned int *v50; // [rsp+B0h] [rbp-C0h] BYREF
  int v51; // [rsp+B8h] [rbp-B8h]
  char v52; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v53; // [rsp+E8h] [rbp-88h]
  __int64 v54; // [rsp+F0h] [rbp-80h]
  __int64 v55; // [rsp+100h] [rbp-70h]
  __int64 v56; // [rsp+108h] [rbp-68h]
  void *v57; // [rsp+130h] [rbp-40h]

  sub_23D0AB0((__int64)&v50, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v2 = *(__int64 **)(a2 - 8);
  else
    v2 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v3 = sub_246F3F0(a1, *v2);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(_QWORD *)(a2 - 8);
  else
    v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v38 = v3;
  v5 = sub_246F3F0(a1, *(_QWORD *)(v4 + 32));
  v49 = 257;
  v6 = *(_QWORD *)(a2 - 64);
  v42 = v5;
  v7 = (_BYTE *)sub_AD62B0(*(_QWORD *)(v6 + 8));
  v8 = sub_A825B0(&v50, (_BYTE *)v6, v7, (__int64)v48);
  v9 = *(_QWORD *)(a2 - 32);
  v49 = 257;
  v10 = v8;
  v11 = (_BYTE *)sub_AD62B0(*(_QWORD *)(v9 + 8));
  v12 = sub_A825B0(&v50, (_BYTE *)v9, v11, (__int64)v48);
  v13 = (_BYTE *)v38;
  v14 = (_BYTE *)v12;
  if ( *(_QWORD *)(v10 + 8) != *(_QWORD *)(v38 + 8) )
  {
    v37 = (_BYTE *)v38;
    v49 = 257;
    v15 = *(__int64 ***)(v38 + 8);
    v39 = v12;
    v16 = sub_BCB060(*(_QWORD *)(v10 + 8));
    v17 = sub_BCB060((__int64)v15);
    v18 = sub_24633A0((__int64 *)&v50, (unsigned int)(v16 <= v17) + 38, v10, v15, (__int64)v48, 0, v46[0], 0);
    v49 = 257;
    v10 = v18;
    v19 = *(__int64 ***)(v42 + 8);
    v20 = sub_BCB060(*(_QWORD *)(v39 + 8));
    v21 = sub_BCB060((__int64)v19);
    v22 = sub_24633A0((__int64 *)&v50, (unsigned int)(v20 <= v21) + 38, v39, v19, (__int64)v48, 0, v46[0], 0);
    v13 = v37;
    v14 = (_BYTE *)v22;
  }
  v23 = (_BYTE *)v42;
  v49 = 257;
  v24 = (_BYTE *)v42;
  v40 = v14;
  v43 = v13;
  v25 = sub_A82350(&v50, v13, v24, (__int64)v48);
  v49 = 257;
  v26 = (unsigned __int8 *)v25;
  v27 = sub_A82350(&v50, (_BYTE *)v10, v23, (__int64)v48);
  v28 = v44;
  v49 = 257;
  v29 = sub_A82350(&v50, v43, v40, (__int64)v48);
  v44[0] = v26;
  v45[0] = v29;
  v44[1] = v27;
  do
  {
    while ( 1 )
    {
      v31 = (unsigned __int8 *)v28[1];
      v47 = 257;
      v32 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v55 + 16LL);
      if ( v32 != sub_9202E0 )
      {
        v30 = v32(v55, 29u, v26, v31);
        goto LABEL_11;
      }
      if ( *v26 > 0x15u || *v31 > 0x15u )
        break;
      if ( (unsigned __int8)sub_AC47B0(29) )
        v30 = sub_AD5570(29, (__int64)v26, v31, 0, 0);
      else
        v30 = sub_AABE40(0x1Du, v26, v31);
LABEL_11:
      if ( !v30 )
        break;
      v26 = (unsigned __int8 *)v30;
LABEL_13:
      if ( v45 == ++v28 )
        goto LABEL_19;
    }
    v49 = 257;
    v26 = (unsigned __int8 *)sub_B504D0(29, (__int64)v26, (__int64)v31, (__int64)v48, 0, 0);
    (*(void (__fastcall **)(__int64, unsigned __int8 *, int *, __int64, __int64))(*(_QWORD *)v56 + 16LL))(
      v56,
      v26,
      v46,
      v53,
      v54);
    v33 = v50;
    v34 = &v50[4 * v51];
    if ( v50 == v34 )
      goto LABEL_13;
    do
    {
      v35 = *((_QWORD *)v33 + 1);
      v36 = *v33;
      v33 += 4;
      sub_B99FD0((__int64)v26, v36, v35);
    }
    while ( v34 != v33 );
    ++v28;
  }
  while ( v45 != v28 );
LABEL_19:
  sub_246EF60(a1, a2, (__int64)v26);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
    sub_2477350(a1, a2);
  nullsub_61();
  v57 = &unk_49DA100;
  nullsub_63();
  if ( v50 != (unsigned int *)&v52 )
    _libc_free((unsigned __int64)v50);
}
