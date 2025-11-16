// Function: sub_2987B70
// Address: 0x2987b70
//
__int64 __fastcall sub_2987B70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // [rsp+18h] [rbp-188h]
  __int64 v19; // [rsp+20h] [rbp-180h] BYREF
  _QWORD *v20; // [rsp+28h] [rbp-178h]
  __int64 v21; // [rsp+30h] [rbp-170h]
  __int64 v22; // [rsp+38h] [rbp-168h]
  _QWORD v23[2]; // [rsp+40h] [rbp-160h] BYREF
  __int64 v24; // [rsp+50h] [rbp-150h] BYREF
  int *v25; // [rsp+58h] [rbp-148h]
  __int64 v26; // [rsp+60h] [rbp-140h]
  __int64 v27; // [rsp+68h] [rbp-138h]
  int v28; // [rsp+70h] [rbp-130h] BYREF
  __int64 v29; // [rsp+78h] [rbp-128h]
  __int64 v30; // [rsp+80h] [rbp-120h]
  __int64 v31; // [rsp+88h] [rbp-118h]
  int v32; // [rsp+90h] [rbp-110h]
  __int64 v33; // [rsp+98h] [rbp-108h]
  __int64 v34; // [rsp+A0h] [rbp-100h]
  __int64 v35; // [rsp+A8h] [rbp-F8h]
  int v36; // [rsp+B0h] [rbp-F0h]
  __int64 v37; // [rsp+B8h] [rbp-E8h]
  __int64 v38; // [rsp+C0h] [rbp-E0h]
  __int64 v39; // [rsp+C8h] [rbp-D8h]
  __int64 v40; // [rsp+D0h] [rbp-D0h]
  __int64 v41; // [rsp+D8h] [rbp-C8h]
  __int64 v42; // [rsp+E0h] [rbp-C0h]
  int v43; // [rsp+E8h] [rbp-B8h]
  __int64 v44; // [rsp+F0h] [rbp-B0h]
  __int64 v45; // [rsp+F8h] [rbp-A8h]
  __int64 v46; // [rsp+100h] [rbp-A0h]
  int v47; // [rsp+108h] [rbp-98h]
  _QWORD *v48; // [rsp+110h] [rbp-90h]
  __int64 v49; // [rsp+118h] [rbp-88h]
  _QWORD v50[3]; // [rsp+120h] [rbp-80h] BYREF
  int v51; // [rsp+138h] [rbp-68h]
  __int64 v52; // [rsp+140h] [rbp-60h]
  __int64 v53; // [rsp+148h] [rbp-58h]
  __int64 v54; // [rsp+150h] [rbp-50h]
  __int64 v55; // [rsp+158h] [rbp-48h]
  __int64 v56; // [rsp+160h] [rbp-40h]
  __int64 v57; // [rsp+168h] [rbp-38h]

  v6 = sub_B2BEC0(a3);
  v18 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v19 = v6;
  v21 = sub_BC1CD0(a4, &unk_4F881D0, a3) + 8;
  v22 = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  v20 = (_QWORD *)(v18 + 8);
  v48 = v50;
  v23[1] = v23;
  v23[0] = v23;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v49 = 0;
  memset(v50, 0, sizeof(v50));
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  LOBYTE(v6) = sub_29851F0((__int64)&v19);
  sub_297D250((__int64)&v19);
  if ( (_BYTE)v6 )
  {
    v25 = &v28;
    v23[0] = &unk_4F82408;
    v20 = v23;
    v21 = 0x100000002LL;
    LODWORD(v22) = 0;
    BYTE4(v22) = 1;
    v24 = 0;
    v26 = 2;
    LODWORD(v27) = 0;
    BYTE4(v27) = 1;
    v19 = 1;
    sub_297BF20((__int64)&v19, (__int64)&unk_4F81450, (__int64)&unk_4F82408, v7, v8, (__int64)&v28);
    sub_297BF20((__int64)&v19, (__int64)&unk_4F881D0, v10, v11, v12, v13);
    sub_297BF20((__int64)&v19, (__int64)&unk_4F89C30, v14, v15, v16, v17);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v23, (__int64)&v19);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)&v28, (__int64)&v24);
    if ( !BYTE4(v27) )
      _libc_free((unsigned __int64)v25);
    if ( !BYTE4(v22) )
      _libc_free((unsigned __int64)v20);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
