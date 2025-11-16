// Function: sub_1FEBC40
// Address: 0x1febc40
//
__int64 __fastcall sub_1FEBC40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        __int64 a8)
{
  __int64 v8; // r15
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r14
  __int64 v14; // rax
  char v15; // dl
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // r15
  char v22; // di
  __int64 v23; // rax
  unsigned int v24; // eax
  char v25; // di
  unsigned int v26; // eax
  char v27; // di
  __int64 v28; // r14
  __int64 v29; // rax
  int v30; // eax
  _QWORD *v31; // rdi
  bool v32; // cf
  __int64 v33; // rsi
  __int64 v34; // rax
  unsigned int v35; // edx
  _QWORD *v36; // rdi
  unsigned int v38; // eax
  __int128 v39; // [rsp-60h] [rbp-130h]
  unsigned int v41; // [rsp+8h] [rbp-C8h]
  int v42; // [rsp+8h] [rbp-C8h]
  int v43; // [rsp+Ch] [rbp-C4h]
  _QWORD *v44; // [rsp+10h] [rbp-C0h]
  __int64 v45; // [rsp+18h] [rbp-B8h]
  unsigned int v47; // [rsp+28h] [rbp-A8h]
  unsigned int v48; // [rsp+2Ch] [rbp-A4h]
  __int64 v49; // [rsp+50h] [rbp-80h] BYREF
  __int64 v50; // [rsp+58h] [rbp-78h]
  __int128 v51; // [rsp+60h] [rbp-70h] BYREF
  __int64 v52; // [rsp+70h] [rbp-60h]
  __int64 v53; // [rsp+80h] [rbp-50h] BYREF
  __int64 v54; // [rsp+88h] [rbp-48h]
  __int64 v55; // [rsp+90h] [rbp-40h]

  v8 = 16LL * (unsigned int)a3;
  v11 = *(_QWORD *)(a1 + 16);
  v50 = a5;
  v12 = *(_QWORD *)(v11 + 32);
  v49 = a4;
  v13 = sub_1E0A0C0(v12);
  v14 = v8 + *(_QWORD *)(a2 + 40);
  v15 = *(_BYTE *)v14;
  v54 = *(_QWORD *)(v14 + 8);
  v16 = *(_QWORD *)(a1 + 16);
  LOBYTE(v53) = v15;
  v17 = sub_1F58E60((__int64)&v53, *(_QWORD **)(v16 + 48));
  v47 = sub_15AAE50(v13, v17);
  v44 = sub_1D29C20(*(_QWORD **)(a1 + 16), (unsigned int)v49, v50, v47, v18, v19);
  v45 = v20;
  sub_1E341E0((__int64)&v51, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL), *((_DWORD *)v44 + 21), 0);
  v21 = *(_QWORD *)(a2 + 40) + v8;
  v22 = *(_BYTE *)v21;
  v23 = *(_QWORD *)(v21 + 8);
  LOBYTE(v53) = v22;
  v54 = v23;
  if ( v22 )
  {
    v24 = sub_1FEB8F0(v22);
    v25 = v49;
    v41 = v24;
    if ( (_BYTE)v49 )
      goto LABEL_3;
LABEL_10:
    v38 = sub_1F58D40((__int64)&v49);
    v27 = a7;
    v48 = v38;
    if ( (_BYTE)a7 )
      goto LABEL_4;
    goto LABEL_11;
  }
  v25 = v49;
  v41 = sub_1F58D40((__int64)&v53);
  if ( !(_BYTE)v49 )
    goto LABEL_10;
LABEL_3:
  v26 = sub_1FEB8F0(v25);
  v27 = a7;
  v48 = v26;
  if ( (_BYTE)a7 )
  {
LABEL_4:
    v43 = sub_1FEB8F0(v27);
    goto LABEL_5;
  }
LABEL_11:
  v43 = sub_1F58D40((__int64)&a7);
LABEL_5:
  v28 = sub_1F58E60((__int64)&a7, *(_QWORD **)(*(_QWORD *)(a1 + 16) + 48LL));
  v29 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL));
  v30 = sub_15AAE50(v29, v28);
  v31 = *(_QWORD **)(a1 + 16);
  v53 = 0;
  v32 = v48 < v41;
  v42 = v30;
  v33 = (__int64)(v31 + 11);
  v54 = 0;
  v55 = 0;
  if ( v32 )
    v34 = sub_1D2C750(v31, v33, 0, a6, a2, a3, (__int64)v44, v45, v51, v52, v49, v50, v47, 0, (__int64)&v53);
  else
    v34 = sub_1D2BF40(v31, v33, 0, a6, a2, a3, (__int64)v44, v45, v51, v52, v47, 0, (__int64)&v53);
  v53 = 0;
  v54 = 0;
  v36 = *(_QWORD **)(a1 + 16);
  v55 = 0;
  if ( v43 == v48 )
    return sub_1D2B730(v36, a7, a8, a6, v34, v35, (__int64)v44, v45, v51, v52, v42, 0, (__int64)&v53, 0);
  *((_QWORD *)&v39 + 1) = v35;
  *(_QWORD *)&v39 = v34;
  return sub_1D2B810(v36, 1u, a6, a7, a8, v42, v39, (__int64)v44, v45, v51, v52, v49, v50, 0, (__int64)&v53);
}
