// Function: sub_32F45B0
// Address: 0x32f45b0
//
__int64 __fastcall sub_32F45B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r10
  __int64 v8; // r13
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned __int16 v14; // dx
  __int64 v15; // rax
  bool v16; // zf
  __int64 v18; // rax
  _QWORD *v19; // r10
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // eax
  int v24; // ecx
  int v25; // eax
  bool v26; // al
  __int64 v27; // rdx
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned int v34; // eax
  __int64 v35; // rax
  __int64 v36; // r8
  __int128 v37; // rax
  int v38; // r9d
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // [rsp+10h] [rbp-C0h]
  __int64 v42; // [rsp+18h] [rbp-B8h]
  int v43; // [rsp+18h] [rbp-B8h]
  __int64 v44; // [rsp+18h] [rbp-B8h]
  __int64 v46; // [rsp+20h] [rbp-B0h]
  _QWORD *v47; // [rsp+28h] [rbp-A8h]
  _QWORD *v48; // [rsp+28h] [rbp-A8h]
  unsigned __int16 v49[4]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v50; // [rsp+38h] [rbp-98h]
  __int64 v51; // [rsp+40h] [rbp-90h] BYREF
  int v52; // [rsp+48h] [rbp-88h]
  __int64 v53; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v54; // [rsp+58h] [rbp-78h]
  __int64 *v55; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v56; // [rsp+68h] [rbp-68h]
  __int64 v57[2]; // [rsp+70h] [rbp-60h] BYREF
  _QWORD v58[2]; // [rsp+80h] [rbp-50h] BYREF
  __int64 v59; // [rsp+90h] [rbp-40h] BYREF
  __int64 v60; // [rsp+98h] [rbp-38h]

  v6 = a1;
  v8 = a6;
  v11 = a5;
  v12 = *(_QWORD *)(a4 + 48) + 16LL * (unsigned int)a5;
  v42 = a2;
  v13 = *(_QWORD *)(a6 + 80);
  v14 = *(_WORD *)v12;
  v15 = *(_QWORD *)(v12 + 8);
  v41 = a4;
  v49[0] = v14;
  v50 = v15;
  v51 = v13;
  if ( v13 )
  {
    sub_B96E90((__int64)&v51, v13, 1);
    v11 = a5;
    v6 = a1;
  }
  v16 = *(_DWORD *)(a2 + 24) == 51;
  v52 = *(_DWORD *)(v8 + 72);
  if ( v16 || *(_DWORD *)(a4 + 24) == 51 )
  {
    v8 = sub_3400BD0(*v6, 0, (unsigned int)&v51, *(_DWORD *)v49, v50, 0, 0);
    goto LABEL_6;
  }
  v47 = v6;
  v18 = sub_32B51D0(v6, 1u, a2, a3, a4, v11, (__int64)&v51);
  v19 = v47;
  if ( v18 )
  {
    v8 = v18;
    goto LABEL_6;
  }
  if ( *(_DWORD *)(a4 + 24) != 56 )
  {
    if ( *(_DWORD *)(a2 + 24) != 56 )
    {
LABEL_13:
      v8 = 0;
      goto LABEL_6;
    }
    v42 = a4;
    v41 = a2;
  }
  if ( *(_DWORD *)(v42 + 24) != 192 )
    goto LABEL_13;
  if ( v49[0] )
  {
    if ( (unsigned __int16)(v49[0] - 2) > 7u )
      goto LABEL_13;
  }
  else
  {
    v26 = sub_30070A0((__int64)v49);
    v19 = v47;
    if ( !v26 )
      goto LABEL_13;
  }
  v48 = v19;
  v59 = sub_2D5B750(v49);
  v60 = v20;
  if ( (unsigned __int64)sub_CA1930(&v59) > 0x40 )
    goto LABEL_13;
  v21 = *(_QWORD *)(v41 + 56);
  if ( !v21 )
    goto LABEL_13;
  if ( *(_QWORD *)(v21 + 32) )
    goto LABEL_13;
  v22 = *(_QWORD *)(*(_QWORD *)(v41 + 40) + 40LL);
  v23 = *(_DWORD *)(v22 + 24);
  if ( v23 != 11 && v23 != 35 )
    goto LABEL_13;
  v24 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v42 + 40) + 40LL) + 24LL);
  if ( v24 != 11 && v24 != 35 )
    goto LABEL_13;
  v46 = *(_QWORD *)(*(_QWORD *)(v42 + 40) + 40LL);
  sub_9865C0((__int64)&v53, *(_QWORD *)(v22 + 96) + 24LL);
  sub_9865C0((__int64)&v55, *(_QWORD *)(v46 + 96) + 24LL);
  v25 = sub_969260((__int64)&v53);
  if ( v54 + 1 - v25 > 0x40
    || (v59 = sub_2D5B750(v49), v60 = v27, v28 = sub_CA1930(&v59), !sub_986EE0((__int64)&v55, v28))
    || (v29 = sub_325F4E0(v53, v54),
        (*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v30 + 1320LL))(v30, v29)) )
  {
LABEL_26:
    sub_969240((__int64 *)&v55);
    sub_969240(&v53);
    goto LABEL_13;
  }
  LODWORD(v31) = (_DWORD)v55;
  if ( v56 > 0x40 )
    v31 = *v55;
  v43 = v31;
  v32 = sub_2D5B750(v49);
  v60 = v33;
  v59 = v32;
  v34 = sub_CA1930(&v59);
  sub_109DDE0((__int64)v57, v34, v43);
  if ( !(unsigned __int8)sub_33DD210(
                           *v48,
                           *(_QWORD *)(*(_QWORD *)(v41 + 40) + 40LL),
                           *(_QWORD *)(*(_QWORD *)(v41 + 40) + 48LL),
                           v57,
                           0)
    || (sub_325F510(&v53, v57),
        v35 = sub_325F4E0(v53, v54),
        !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v36 + 1320LL))(v36, v35)) )
  {
    sub_969240(v57);
    goto LABEL_26;
  }
  sub_3285E70((__int64)v58, v41);
  v44 = *v48;
  *(_QWORD *)&v37 = sub_34007B0(*v48, (unsigned int)&v53, (unsigned int)&v51, *(_DWORD *)v49, v50, 0, 0);
  v39 = sub_3406EB0(v44, 56, (unsigned int)v58, *(_DWORD *)v49, v50, v38, *(_OWORD *)*(_QWORD *)(v41 + 40), v37);
  v60 = v40;
  v59 = v39;
  sub_32EB790((__int64)v48, v41, &v59, 1, 1);
  sub_9C6650(v58);
  sub_969240(v57);
  sub_969240((__int64 *)&v55);
  sub_969240(&v53);
LABEL_6:
  if ( v51 )
    sub_B91220((__int64)&v51, v51);
  return v8;
}
