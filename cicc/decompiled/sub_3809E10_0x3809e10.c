// Function: sub_3809E10
// Address: 0x3809e10
//
unsigned __int64 __fastcall sub_3809E10(__int64 *a1, unsigned __int64 a2, int a3)
{
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v6; // rax
  unsigned __int16 v7; // si
  __int64 v8; // r8
  __int64 v9; // rax
  unsigned int v10; // ebx
  int v11; // eax
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 *v16; // rax
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 *v19; // rax
  __int64 v20; // rsi
  _WORD *v21; // r10
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r11
  __int64 v29; // rdx
  unsigned int v30; // eax
  __int64 (__fastcall *v31)(__int64, __int64, unsigned int); // rdx
  __int64 v32; // [rsp+0h] [rbp-E0h]
  __int64 v33; // [rsp+8h] [rbp-D8h]
  _WORD *v34; // [rsp+18h] [rbp-C8h]
  __int64 (__fastcall *v35)(__int64, __int64, unsigned int); // [rsp+20h] [rbp-C0h]
  char v36; // [rsp+2Fh] [rbp-B1h]
  unsigned __int64 v37; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v38; // [rsp+38h] [rbp-A8h]
  __int16 v39; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+48h] [rbp-98h]
  __int64 v41; // [rsp+50h] [rbp-90h] BYREF
  int v42; // [rsp+58h] [rbp-88h]
  unsigned __int64 v43[4]; // [rsp+60h] [rbp-80h] BYREF
  __int16 *v44; // [rsp+80h] [rbp-60h] BYREF
  __int64 v45; // [rsp+88h] [rbp-58h]
  __int64 (__fastcall *v46)(__int64, __int64, unsigned int); // [rsp+90h] [rbp-50h]
  __int64 v47; // [rsp+98h] [rbp-48h]
  __int64 v48; // [rsp+A0h] [rbp-40h]

  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = a1[1];
  if ( v5 == sub_2D56A50 )
  {
    HIWORD(v10) = 0;
    sub_2FE6CC0((__int64)&v44, *a1, *(_QWORD *)(v9 + 64), v7, v8);
    LOWORD(v10) = v45;
    v35 = v46;
  }
  else
  {
    v30 = v5(*a1, *(_QWORD *)(v9 + 64), v7, v8);
    v35 = v31;
    v10 = v30;
  }
  v11 = *(_DWORD *)(a2 + 24);
  v12 = *(_QWORD *)(a2 + 40);
  if ( v11 > 239 )
  {
    if ( (unsigned int)(v11 - 242) > 1 )
    {
LABEL_6:
      v13 = sub_3805E70((__int64)a1, *(_QWORD *)v12, *(_QWORD *)(v12 + 8));
      v36 = 0;
      v14 = 0;
      v37 = v13;
      v38 = v15;
      v16 = *(__int64 **)(a2 + 40);
      v17 = 0;
      v18 = 0;
      goto LABEL_7;
    }
  }
  else if ( v11 <= 237 && (unsigned int)(v11 - 101) > 0x2F )
  {
    goto LABEL_6;
  }
  v36 = 1;
  v37 = sub_3805E70((__int64)a1, *(_QWORD *)(v12 + 40), *(_QWORD *)(v12 + 48));
  v16 = *(__int64 **)(a2 + 40);
  v38 = v29;
  v18 = 5;
  v14 = *v16;
  v17 = v16[1];
LABEL_7:
  v19 = &v16[v18];
  v20 = *(_QWORD *)(a2 + 80);
  v21 = (_WORD *)*a1;
  LOBYTE(v48) = 20;
  v22 = *v19;
  v23 = *((unsigned int *)v19 + 2);
  v45 = 1;
  v44 = &v39;
  v24 = *(_QWORD *)(v22 + 48) + 16 * v23;
  LOWORD(v22) = *(_WORD *)v24;
  v40 = *(_QWORD *)(v24 + 8);
  v25 = *(_QWORD *)(a2 + 48);
  v39 = v22;
  LOWORD(v22) = *(_WORD *)v25;
  v26 = *(_QWORD *)(v25 + 8);
  v41 = v20;
  LOWORD(v46) = v22;
  v47 = v26;
  if ( v20 )
  {
    v32 = v14;
    v33 = v17;
    v34 = v21;
    sub_B96E90((__int64)&v41, v20, 1);
    v14 = v32;
    v17 = v33;
    v21 = v34;
  }
  v27 = a1[1];
  v42 = *(_DWORD *)(a2 + 72);
  sub_3494590(
    (__int64)v43,
    v21,
    v27,
    a3,
    v10,
    v35,
    (__int64)&v37,
    1u,
    (__int64)v44,
    v45,
    (unsigned int)v46,
    v47,
    v48,
    (__int64)&v41,
    v14,
    v17);
  if ( v41 )
    sub_B91220((__int64)&v41, v41);
  if ( !v36 )
    return v43[0];
  sub_3760E70((__int64)a1, a2, 1, v43[2], v43[3]);
  sub_3760E70((__int64)a1, a2, 0, v43[0], v43[1]);
  return 0;
}
