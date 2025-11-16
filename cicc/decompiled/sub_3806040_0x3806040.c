// Function: sub_3806040
// Address: 0x3806040
//
__int64 __fastcall sub_3806040(__int64 *a1, unsigned __int64 a2, int a3)
{
  int v4; // eax
  bool v5; // bl
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v7; // rax
  unsigned __int16 v8; // si
  __int64 v9; // r8
  __int64 v10; // rax
  unsigned int v11; // r13d
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int); // r9
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int); // r9
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // r10
  __int64 v20; // r11
  __int64 *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rsi
  unsigned __int64 v31; // rax
  __int64 v32; // rdx
  __int64 (__fastcall *v33)(__int64, __int64, unsigned int); // rdx
  __int64 (__fastcall *v34)(__int64, __int64, unsigned int); // [rsp+8h] [rbp-D8h]
  __int64 v35; // [rsp+10h] [rbp-D0h]
  __int64 v36; // [rsp+18h] [rbp-C8h]
  __int64 (__fastcall *v38)(__int64, __int64, unsigned int); // [rsp+28h] [rbp-B8h]
  _WORD *v39; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v40; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v41; // [rsp+38h] [rbp-A8h]
  __int16 v42; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v43; // [rsp+48h] [rbp-98h]
  __int64 v44; // [rsp+50h] [rbp-90h] BYREF
  int v45; // [rsp+58h] [rbp-88h]
  _QWORD v46[4]; // [rsp+60h] [rbp-80h] BYREF
  __int16 *v47; // [rsp+80h] [rbp-60h] BYREF
  __int64 v48; // [rsp+88h] [rbp-58h]
  __int64 (__fastcall *v49)(__int64, __int64, unsigned int); // [rsp+90h] [rbp-50h]
  __int64 v50; // [rsp+98h] [rbp-48h]
  __int64 v51; // [rsp+A0h] [rbp-40h]

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
    sub_2FE6CC0((__int64)&v47, *a1, *(_QWORD *)(v10 + 64), v8, v9);
    LOWORD(v11) = v48;
    v12 = v49;
  }
  else
  {
    v11 = v6(*a1, *(_QWORD *)(v10 + 64), v8, v9);
    v12 = v33;
  }
  v13 = *(_QWORD *)(a2 + 40);
  v38 = v12;
  if ( v5 )
  {
    v14 = sub_3805E70((__int64)a1, *(_QWORD *)(v13 + 40), *(_QWORD *)(v13 + 48));
    v15 = v38;
    v40 = v14;
    v16 = *(__int64 **)(a2 + 40);
    v41 = v17;
    v18 = 5;
    v19 = *v16;
    v20 = v16[1];
  }
  else
  {
    v31 = sub_3805E70((__int64)a1, *(_QWORD *)v13, *(_QWORD *)(v13 + 8));
    v15 = v38;
    v19 = 0;
    v40 = v31;
    v41 = v32;
    v16 = *(__int64 **)(a2 + 40);
    v20 = 0;
    v18 = 0;
  }
  v21 = &v16[v18];
  v22 = *(_QWORD *)(a2 + 80);
  LOBYTE(v51) = 20;
  v23 = *v21;
  v24 = *((unsigned int *)v21 + 2);
  v47 = &v42;
  v48 = 1;
  v25 = *(_QWORD *)(v23 + 48) + 16 * v24;
  LOWORD(v23) = *(_WORD *)v25;
  v43 = *(_QWORD *)(v25 + 8);
  v26 = *(_QWORD *)(a2 + 48);
  v42 = v23;
  LOWORD(v23) = *(_WORD *)v26;
  v27 = *(_QWORD *)(v26 + 8);
  v44 = v22;
  v50 = v27;
  v28 = *a1;
  LOWORD(v49) = v23;
  v39 = (_WORD *)v28;
  if ( v22 )
  {
    v34 = v15;
    v35 = v19;
    v36 = v20;
    sub_B96E90((__int64)&v44, v22, 1);
    v15 = v34;
    v19 = v35;
    v20 = v36;
  }
  v29 = a1[1];
  v45 = *(_DWORD *)(a2 + 72);
  sub_3494590(
    (__int64)v46,
    v39,
    v29,
    a3,
    v11,
    v15,
    (__int64)&v40,
    1u,
    (__int64)v47,
    v48,
    (unsigned int)v49,
    v50,
    v51,
    (__int64)&v44,
    v19,
    v20);
  if ( v44 )
    sub_B91220((__int64)&v44, v44);
  if ( v5 )
    sub_3760E70((__int64)a1, a2, 1, v46[2], v46[3]);
  return v46[0];
}
