// Function: sub_38062F0
// Address: 0x38062f0
//
__int64 __fastcall sub_38062F0(__int64 *a1, unsigned __int64 a2, int a3)
{
  int v4; // eax
  bool v5; // bl
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v7; // rax
  unsigned __int16 v8; // si
  __int64 v9; // r8
  __int64 v10; // rax
  unsigned int v11; // r14d
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int); // r9
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 *v17; // rcx
  __int64 (__fastcall *v18)(__int64, __int64, unsigned int); // r9
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r10
  __int64 v22; // rdx
  __int64 v23; // r11
  __int64 *v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rax
  __int16 v31; // dx
  __int64 v32; // rax
  __int16 *v33; // rax
  __int16 v34; // dx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rsi
  __int64 v39; // rax
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  __int64 v42; // rdx
  __int64 (__fastcall *v43)(__int64, __int64, unsigned int); // rdx
  __int64 (__fastcall *v44)(__int64, __int64, unsigned int); // [rsp+8h] [rbp-F8h]
  __int64 v45; // [rsp+10h] [rbp-F0h]
  __int64 v46; // [rsp+18h] [rbp-E8h]
  __int64 (__fastcall *v48)(__int64, __int64, unsigned int); // [rsp+28h] [rbp-D8h]
  _WORD *v49; // [rsp+28h] [rbp-D8h]
  __int64 v50; // [rsp+30h] [rbp-D0h] BYREF
  int v51; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v52; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v54; // [rsp+50h] [rbp-B0h]
  __int64 v55; // [rsp+58h] [rbp-A8h]
  __int16 v56; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v57; // [rsp+68h] [rbp-98h]
  __int16 v58; // [rsp+70h] [rbp-90h]
  __int64 v59; // [rsp+78h] [rbp-88h]
  _QWORD v60[4]; // [rsp+80h] [rbp-80h] BYREF
  __int16 *v61; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v62; // [rsp+A8h] [rbp-58h]
  __int64 (__fastcall *v63)(__int64, __int64, unsigned int); // [rsp+B0h] [rbp-50h]
  __int64 v64; // [rsp+B8h] [rbp-48h]
  __int64 v65; // [rsp+C0h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 24);
  if ( v4 > 239 )
  {
    v5 = (unsigned int)(v4 - 242) <= 1;
  }
  else
  {
    v5 = 1;
    if ( v4 <= 237 )
      v5 = (unsigned int)(v4 - 101) <= 0x2F;
  }
  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  v10 = a1[1];
  if ( v6 == sub_2D56A50 )
  {
    HIWORD(v11) = 0;
    sub_2FE6CC0((__int64)&v61, *a1, *(_QWORD *)(v10 + 64), v8, v9);
    LOWORD(v11) = v62;
    v12 = v63;
  }
  else
  {
    v11 = v6(*a1, *(_QWORD *)(v10 + 64), v8, v9);
    v12 = v43;
  }
  v13 = *(_QWORD *)(a2 + 40);
  v48 = v12;
  if ( v5 )
  {
    v52 = sub_3805E70((__int64)a1, *(_QWORD *)(v13 + 40), *(_QWORD *)(v13 + 48));
    v14 = *(_QWORD *)(a2 + 40);
    v53 = v15;
    v16 = sub_3805E70((__int64)a1, *(_QWORD *)(v14 + 80), *(_QWORD *)(v14 + 88));
    v17 = *(__int64 **)(a2 + 40);
    v18 = v48;
    v54 = v16;
    v19 = 5;
    v55 = v20;
    v21 = *v17;
    v22 = 10;
    v23 = v17[1];
  }
  else
  {
    v52 = sub_3805E70((__int64)a1, *(_QWORD *)v13, *(_QWORD *)(v13 + 8));
    v39 = *(_QWORD *)(a2 + 40);
    v53 = v40;
    v41 = sub_3805E70((__int64)a1, *(_QWORD *)(v39 + 40), *(_QWORD *)(v39 + 48));
    v17 = *(__int64 **)(a2 + 40);
    v21 = 0;
    v54 = v41;
    v18 = v48;
    v55 = v42;
    v19 = 0;
    v23 = 0;
    v22 = 5;
  }
  v24 = &v17[v19];
  LOBYTE(v65) = 20;
  v25 = *v24;
  v26 = *((unsigned int *)v24 + 2);
  v62 = 2;
  v27 = *(_QWORD *)(v25 + 48) + 16 * v26;
  LOWORD(v25) = *(_WORD *)v27;
  v28 = *(_QWORD *)(v27 + 8);
  v56 = v25;
  v29 = *(_QWORD *)(a2 + 80);
  v57 = v28;
  v30 = *(_QWORD *)(v17[v22] + 48) + 16LL * LODWORD(v17[v22 + 1]);
  v31 = *(_WORD *)v30;
  v32 = *(_QWORD *)(v30 + 8);
  v61 = &v56;
  v59 = v32;
  v33 = *(__int16 **)(a2 + 48);
  v58 = v31;
  v34 = *v33;
  v35 = *((_QWORD *)v33 + 1);
  v50 = v29;
  v64 = v35;
  v36 = *a1;
  LOWORD(v63) = v34;
  v49 = (_WORD *)v36;
  if ( v29 )
  {
    v44 = v18;
    v45 = v21;
    v46 = v23;
    sub_B96E90((__int64)&v50, v29, 1);
    v18 = v44;
    v21 = v45;
    v23 = v46;
  }
  v37 = a1[1];
  v51 = *(_DWORD *)(a2 + 72);
  sub_3494590(
    (__int64)v60,
    v49,
    v37,
    a3,
    v11,
    v18,
    (__int64)&v52,
    2u,
    (__int64)v61,
    v62,
    (unsigned int)v63,
    v64,
    v65,
    (__int64)&v50,
    v21,
    v23);
  if ( v50 )
    sub_B91220((__int64)&v50, v50);
  if ( v5 )
    sub_3760E70((__int64)a1, a2, 1, v60[2], v60[3]);
  return v60[0];
}
