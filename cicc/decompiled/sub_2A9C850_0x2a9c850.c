// Function: sub_2A9C850
// Address: 0x2a9c850
//
__int64 __fastcall sub_2A9C850(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r13
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // [rsp+0h] [rbp-590h]
  __int64 v16; // [rsp+8h] [rbp-588h]
  __int64 v17; // [rsp+10h] [rbp-580h]
  char v18; // [rsp+10h] [rbp-580h]
  __int64 v19; // [rsp+18h] [rbp-578h]
  __int64 v20; // [rsp+20h] [rbp-570h] BYREF
  _QWORD *v21; // [rsp+28h] [rbp-568h]
  __int64 v22; // [rsp+30h] [rbp-560h]
  __int64 v23; // [rsp+38h] [rbp-558h]
  _QWORD v24[2]; // [rsp+40h] [rbp-550h] BYREF
  __int64 v25; // [rsp+50h] [rbp-540h] BYREF
  int *v26; // [rsp+58h] [rbp-538h]
  __int64 v27; // [rsp+60h] [rbp-530h]
  int v28; // [rsp+68h] [rbp-528h] BYREF
  char v29; // [rsp+6Ch] [rbp-524h]
  char v30; // [rsp+70h] [rbp-520h] BYREF
  __int64 v31; // [rsp+88h] [rbp-508h]
  __int64 v32; // [rsp+90h] [rbp-500h]
  __int16 v33; // [rsp+98h] [rbp-4F8h]
  __int64 v34; // [rsp+A0h] [rbp-4F0h]
  void **v35; // [rsp+A8h] [rbp-4E8h]
  _QWORD *v36; // [rsp+B0h] [rbp-4E0h]
  __int64 v37; // [rsp+B8h] [rbp-4D8h]
  int v38; // [rsp+C0h] [rbp-4D0h]
  __int16 v39; // [rsp+C4h] [rbp-4CCh]
  char v40; // [rsp+C6h] [rbp-4CAh]
  __int64 v41; // [rsp+C8h] [rbp-4C8h]
  __int64 v42; // [rsp+D0h] [rbp-4C0h]
  void *v43; // [rsp+D8h] [rbp-4B8h] BYREF
  _QWORD v44[2]; // [rsp+E0h] [rbp-4B0h] BYREF
  __int64 v45; // [rsp+F0h] [rbp-4A0h]
  __int64 v46; // [rsp+F8h] [rbp-498h]
  unsigned int v47; // [rsp+100h] [rbp-490h]
  _BYTE *v48; // [rsp+108h] [rbp-488h]
  __int64 v49; // [rsp+110h] [rbp-480h]
  _BYTE v50[1024]; // [rsp+118h] [rbp-478h] BYREF
  __int64 v51; // [rsp+518h] [rbp-78h]
  __int64 v52; // [rsp+520h] [rbp-70h]
  __int64 v53; // [rsp+528h] [rbp-68h]
  __int64 v54; // [rsp+530h] [rbp-60h]
  __int64 v55; // [rsp+538h] [rbp-58h]
  __int64 v56; // [rsp+540h] [rbp-50h]
  __int64 v57; // [rsp+548h] [rbp-48h]
  unsigned int v58; // [rsp+550h] [rbp-40h]

  v6 = a1 + 32;
  v19 = a1 + 80;
  if ( (unsigned __int8)sub_B2D610(a3, 30) )
  {
    *(_QWORD *)(a1 + 8) = v6;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
LABEL_3:
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v8 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v15 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v16 = sub_BC1CD0(a4, &unk_4F881D0, a3);
  v17 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v21 = (_QWORD *)(v8 + 8);
  v20 = a3;
  v22 = sub_BC1CD0(a4, &unk_4F86630, a3) + 8;
  v24[0] = v16 + 8;
  v23 = v15 + 8;
  v24[1] = v17 + 8;
  v25 = sub_B2BEC0(a3);
  v34 = sub_B2BE50(*(_QWORD *)(v16 + 8));
  v39 = 512;
  v26 = &v28;
  v43 = &unk_49DA100;
  v27 = 0x200000000LL;
  v48 = v50;
  v44[0] = &unk_49DA0B0;
  v33 = 0;
  v49 = 0x8000000000LL;
  v35 = &v43;
  v36 = v44;
  v37 = 0;
  v38 = 0;
  v40 = 7;
  v41 = 0;
  v42 = 0;
  v31 = 0;
  v32 = 0;
  v44[1] = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v18 = sub_2A99860((__int64)&v20);
  sub_C7D6A0(v56, 24LL * v58, 8);
  sub_C7D6A0(v52, 8LL * (unsigned int)v54, 8);
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  sub_C7D6A0(v45, 16LL * v47, 8);
  nullsub_61();
  v43 = &unk_49DA100;
  nullsub_63();
  if ( v26 != &v28 )
    _libc_free((unsigned __int64)v26);
  v22 = 0x100000002LL;
  v21 = v24;
  v26 = (int *)&v30;
  LODWORD(v23) = 0;
  BYTE4(v23) = 1;
  v25 = 0;
  v27 = 2;
  v28 = 0;
  v29 = 1;
  v24[0] = &unk_4F82408;
  v20 = 1;
  if ( !v18 )
  {
    *(_QWORD *)(a1 + 8) = v6;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v19;
    goto LABEL_3;
  }
  sub_C8CD80(a1, a1 + 32, (__int64)&v20, v9, v10, v11);
  sub_C8CD80(a1 + 48, v19, (__int64)&v25, v12, v13, v14);
  if ( !v29 )
    _libc_free((unsigned __int64)v26);
  if ( !BYTE4(v23) )
    _libc_free((unsigned __int64)v21);
  return a1;
}
