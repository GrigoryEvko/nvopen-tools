// Function: sub_2478680
// Address: 0x2478680
//
void __fastcall sub_2478680(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // rdx
  unsigned __int64 v5; // r11
  _BYTE *v6; // r10
  __int64 **v7; // r15
  unsigned int v8; // eax
  unsigned __int64 v9; // rax
  __int64 **v10; // r15
  unsigned int v11; // r13d
  unsigned int v12; // eax
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int8 *v15; // r15
  _QWORD *v16; // r12
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int8 *v19; // r13
  __int64 (__fastcall *v20)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  unsigned int *v21; // r13
  unsigned int *v22; // rbx
  __int64 v23; // rdx
  unsigned int v24; // esi
  unsigned __int64 v25; // [rsp+0h] [rbp-170h]
  unsigned __int64 v26; // [rsp+8h] [rbp-168h]
  unsigned __int64 v27; // [rsp+8h] [rbp-168h]
  _BYTE *v28; // [rsp+8h] [rbp-168h]
  unsigned int v29; // [rsp+18h] [rbp-158h]
  _BYTE *v30; // [rsp+18h] [rbp-158h]
  __int64 v32; // [rsp+28h] [rbp-148h]
  __int64 v33; // [rsp+28h] [rbp-148h]
  _QWORD v34[2]; // [rsp+30h] [rbp-140h] BYREF
  _QWORD v35[2]; // [rsp+40h] [rbp-130h] BYREF
  int v36[8]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v37; // [rsp+70h] [rbp-100h]
  _BYTE v38[32]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v39; // [rsp+A0h] [rbp-D0h]
  unsigned int *v40; // [rsp+B0h] [rbp-C0h] BYREF
  int v41; // [rsp+B8h] [rbp-B8h]
  char v42; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v43; // [rsp+E8h] [rbp-88h]
  __int64 v44; // [rsp+F0h] [rbp-80h]
  __int64 v45; // [rsp+100h] [rbp-70h]
  __int64 v46; // [rsp+108h] [rbp-68h]
  void *v47; // [rsp+130h] [rbp-40h]

  sub_23D0AB0((__int64)&v40, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v2 = *(__int64 **)(a2 - 8);
  else
    v2 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v3 = sub_246F3F0(a1, *v2);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(_QWORD *)(a2 - 8);
  else
    v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v32 = sub_246F3F0(a1, *(_QWORD *)(v4 + 32));
  v5 = *(_QWORD *)(a2 - 64);
  v6 = *(_BYTE **)(a2 - 32);
  if ( *(_QWORD *)(v5 + 8) != *(_QWORD *)(v3 + 8) )
  {
    v25 = *(_QWORD *)(a2 - 32);
    v39 = 257;
    v7 = *(__int64 ***)(v3 + 8);
    v26 = v5;
    v29 = sub_BCB060(*(_QWORD *)(v5 + 8));
    v8 = sub_BCB060((__int64)v7);
    v9 = sub_24633A0((__int64 *)&v40, (unsigned int)(v29 <= v8) + 38, v26, v7, (__int64)v38, 0, v36[0], 0);
    v39 = 257;
    v10 = *(__int64 ***)(v32 + 8);
    v27 = v9;
    v11 = sub_BCB060(*(_QWORD *)(v25 + 8));
    v12 = sub_BCB060((__int64)v10);
    v13 = sub_24633A0((__int64 *)&v40, (unsigned int)(v11 <= v12) + 38, v25, v10, (__int64)v38, 0, v36[0], 0);
    v5 = v27;
    v6 = (_BYTE *)v13;
  }
  v39 = 257;
  v28 = v6;
  v30 = (_BYTE *)v5;
  v14 = sub_A82350(&v40, (_BYTE *)v3, (_BYTE *)v32, (__int64)v38);
  v39 = 257;
  v15 = (unsigned __int8 *)v14;
  v33 = sub_A82350(&v40, v30, (_BYTE *)v32, (__int64)v38);
  v16 = v34;
  v39 = 257;
  v17 = sub_A82350(&v40, (_BYTE *)v3, v28, (__int64)v38);
  v34[0] = v15;
  v35[0] = v17;
  v34[1] = v33;
  do
  {
    while ( 1 )
    {
      v19 = (unsigned __int8 *)v16[1];
      v37 = 257;
      v20 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))(*(_QWORD *)v45 + 16LL);
      if ( v20 != sub_9202E0 )
      {
        v18 = v20(v45, 29u, v15, v19);
        goto LABEL_11;
      }
      if ( *v15 > 0x15u || *v19 > 0x15u )
        break;
      if ( (unsigned __int8)sub_AC47B0(29) )
        v18 = sub_AD5570(29, (__int64)v15, v19, 0, 0);
      else
        v18 = sub_AABE40(0x1Du, v15, v19);
LABEL_11:
      if ( !v18 )
        break;
      v15 = (unsigned __int8 *)v18;
LABEL_13:
      if ( v35 == ++v16 )
        goto LABEL_19;
    }
    v39 = 257;
    v15 = (unsigned __int8 *)sub_B504D0(29, (__int64)v15, (__int64)v19, (__int64)v38, 0, 0);
    (*(void (__fastcall **)(__int64, unsigned __int8 *, int *, __int64, __int64))(*(_QWORD *)v46 + 16LL))(
      v46,
      v15,
      v36,
      v43,
      v44);
    v21 = v40;
    v22 = &v40[4 * v41];
    if ( v40 == v22 )
      goto LABEL_13;
    do
    {
      v23 = *((_QWORD *)v21 + 1);
      v24 = *v21;
      v21 += 4;
      sub_B99FD0((__int64)v15, v24, v23);
    }
    while ( v22 != v21 );
    ++v16;
  }
  while ( v35 != v16 );
LABEL_19:
  sub_246EF60(a1, a2, (__int64)v15);
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
    sub_2477350(a1, a2);
  nullsub_61();
  v47 = &unk_49DA100;
  nullsub_63();
  if ( v40 != (unsigned int *)&v42 )
    _libc_free((unsigned __int64)v40);
}
