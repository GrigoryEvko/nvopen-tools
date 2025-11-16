// Function: sub_3807360
// Address: 0x3807360
//
__int64 __fastcall sub_3807360(__int64 *a1, unsigned __int64 a2)
{
  int v4; // eax
  unsigned int v5; // eax
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // r15
  __int16 *v9; // rax
  unsigned __int16 v10; // di
  __int64 v11; // r8
  __int64 v12; // rax
  unsigned int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  unsigned int v19; // ecx
  __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  char *v24; // rbx
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // r15
  _WORD *v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int16 *v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // r15d
  __int16 v38; // si
  unsigned int v40; // eax
  unsigned int v41; // eax
  __int64 (__fastcall *v42)(__int64, __int64, unsigned int); // rdx
  unsigned int v43; // [rsp+8h] [rbp-118h]
  __int64 v44; // [rsp+10h] [rbp-110h]
  __int64 v45; // [rsp+10h] [rbp-110h]
  __int64 v46; // [rsp+18h] [rbp-108h]
  __int64 (__fastcall *v47)(__int64, __int64, unsigned int); // [rsp+20h] [rbp-100h]
  bool v48; // [rsp+2Fh] [rbp-F1h]
  __int64 v49; // [rsp+30h] [rbp-F0h] BYREF
  int v50; // [rsp+38h] [rbp-E8h]
  _QWORD v51[4]; // [rsp+40h] [rbp-E0h] BYREF
  __int16 *v52; // [rsp+60h] [rbp-C0h]
  __int64 v53; // [rsp+68h] [rbp-B8h]
  __int64 v54; // [rsp+70h] [rbp-B0h]
  __int64 v55; // [rsp+78h] [rbp-A8h]
  __int64 v56; // [rsp+80h] [rbp-A0h]
  _QWORD v57[6]; // [rsp+90h] [rbp-90h] BYREF
  __int16 v58; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v59; // [rsp+C8h] [rbp-58h]
  __int64 (__fastcall *v60)(__int64, __int64, unsigned int); // [rsp+D0h] [rbp-50h]
  __int64 v61; // [rsp+D8h] [rbp-48h]
  __int16 v62; // [rsp+E0h] [rbp-40h]
  __int64 v63; // [rsp+E8h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 24);
  if ( v4 > 239 )
  {
    v40 = v4 - 242;
    v6 = v40 < 2 ? 0x28 : 0;
    v7 = v6 + 40;
    v48 = v40 < 2;
    v44 = v6 + 80;
  }
  else if ( v4 > 237 )
  {
    v44 = 120;
    v6 = 40;
    v7 = 80;
    v48 = 1;
  }
  else
  {
    v5 = v4 - 101;
    v6 = v5 < 0x30 ? 0x28 : 0;
    v7 = v6 + 40;
    v48 = v5 < 0x30;
    v44 = v6 + 80;
  }
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v12 = a1[1];
  if ( v8 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v58, *a1, *(_QWORD *)(v12 + 64), v10, v11);
    v13 = (unsigned __int16)v59;
    v47 = v60;
  }
  else
  {
    v41 = v8(*a1, *(_QWORD *)(v12 + 64), v10, v11);
    v47 = v42;
    v13 = v41;
  }
  v43 = v13;
  v57[0] = sub_3805E70((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + v6), *(_QWORD *)(*(_QWORD *)(a2 + 40) + v6 + 8));
  v14 = *(_QWORD *)(a2 + 40);
  v57[1] = v15;
  v57[2] = sub_3805E70((__int64)a1, *(_QWORD *)(v14 + v7), *(_QWORD *)(v14 + v7 + 8));
  v16 = *(_QWORD *)(a2 + 40);
  v57[3] = v17;
  v18 = sub_3805E70((__int64)a1, *(_QWORD *)(v16 + v44), *(_QWORD *)(v16 + v44 + 8));
  v19 = v43;
  v57[4] = v18;
  v57[5] = v20;
  if ( v48 )
  {
    v21 = *(__int64 **)(a2 + 40);
    v22 = *v21;
    v23 = v21[1];
  }
  else
  {
    v22 = 0;
    v21 = *(__int64 **)(a2 + 40);
    v23 = 0;
  }
  v24 = (char *)v21 + v6;
  v52 = &v58;
  v25 = *(_QWORD *)v24;
  v26 = *((unsigned int *)v24 + 2);
  v53 = 3;
  v27 = *(_QWORD *)(a2 + 80);
  v28 = (_WORD *)*a1;
  LOBYTE(v56) = 20;
  v29 = *(_QWORD *)(v25 + 48) + 16 * v26;
  LOWORD(v25) = *(_WORD *)v29;
  v30 = *(_QWORD *)(v29 + 8);
  v58 = v25;
  v59 = v30;
  v31 = *(_QWORD *)(*(__int64 *)((char *)v21 + v7) + 48) + 16LL * *(unsigned int *)((char *)v21 + v7 + 8);
  LOWORD(v25) = *(_WORD *)v31;
  v32 = *(_QWORD *)(v31 + 8);
  LOWORD(v60) = v25;
  v33 = *(__int16 **)(a2 + 48);
  v61 = v32;
  v34 = *(_QWORD *)(*(__int64 *)((char *)v21 + v44) + 48) + 16LL * *(unsigned int *)((char *)v21 + v44 + 8);
  LOWORD(v32) = *(_WORD *)v34;
  v35 = *(_QWORD *)(v34 + 8);
  v62 = v32;
  v63 = v35;
  v36 = *((_QWORD *)v33 + 1);
  LOWORD(v54) = *v33;
  v55 = v36;
  v49 = v27;
  if ( v27 )
  {
    v45 = v22;
    v46 = v23;
    sub_B96E90((__int64)&v49, v27, 1);
    v33 = *(__int16 **)(a2 + 48);
    v19 = v43;
    v22 = v45;
    v23 = v46;
  }
  v37 = 80;
  v50 = *(_DWORD *)(a2 + 72);
  v38 = *v33;
  if ( v38 != 12 )
  {
    v37 = 81;
    if ( v38 != 13 )
    {
      v37 = 82;
      if ( v38 != 14 )
      {
        v37 = 83;
        if ( v38 != 15 )
        {
          v37 = 729;
          if ( v38 == 16 )
            v37 = 84;
        }
      }
    }
  }
  sub_3494590(
    (__int64)v51,
    v28,
    a1[1],
    v37,
    v19,
    v47,
    (__int64)v57,
    3u,
    (__int64)v52,
    v53,
    v54,
    v55,
    v56,
    (__int64)&v49,
    v22,
    v23);
  if ( v49 )
    sub_B91220((__int64)&v49, v49);
  if ( v48 )
    sub_3760E70((__int64)a1, a2, 1, v51[2], v51[3]);
  return v51[0];
}
