// Function: sub_38079C0
// Address: 0x38079c0
//
_QWORD *__fastcall sub_38079C0(__int64 *a1, unsigned __int64 a2)
{
  int v4; // eax
  __int64 v5; // r14
  int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int16 v9; // dx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdi
  _QWORD *v14; // rdi
  __int64 v15; // r8
  unsigned int v16; // ecx
  _QWORD *v17; // r12
  __int64 v19; // rdx
  __int64 (__fastcall *v20)(__int64, __int64, unsigned int, __int64); // r8
  __int16 *v21; // rax
  unsigned __int16 v22; // di
  __int64 v23; // rax
  __int64 (__fastcall *v24)(__int64, __int64, unsigned int); // r9
  __int64 v25; // r8
  __int64 v26; // r14
  unsigned __int64 v27; // rax
  __int64 v28; // r8
  __int64 (__fastcall *v29)(__int64, __int64, unsigned int); // r9
  __int64 *v30; // rax
  __int64 v31; // rdx
  unsigned int *v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // rbx
  _WORD *v35; // r11
  __int64 v36; // rax
  __int16 v37; // si
  __int64 v38; // rax
  __int64 v39; // rax
  __int16 v40; // dx
  __int16 *v41; // rax
  __int16 v42; // dx
  __int64 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // r10
  __int64 v46; // rdi
  _QWORD *v47; // rdi
  __int64 v48; // r8
  unsigned int v49; // ecx
  __int64 (__fastcall *v50)(__int64, __int64, unsigned int); // rdx
  __int64 (__fastcall *v51)(__int64, __int64, unsigned int); // [rsp+0h] [rbp-110h]
  __int64 (__fastcall *v52)(__int64, __int64, unsigned int); // [rsp+8h] [rbp-108h]
  __int64 v53; // [rsp+8h] [rbp-108h]
  __int64 v54; // [rsp+10h] [rbp-100h]
  __int64 v55; // [rsp+10h] [rbp-100h]
  __int64 v56; // [rsp+20h] [rbp-F0h]
  __int64 v57; // [rsp+20h] [rbp-F0h]
  _WORD *v58; // [rsp+20h] [rbp-F0h]
  bool v59; // [rsp+2Fh] [rbp-E1h]
  __int64 v60; // [rsp+30h] [rbp-E0h] BYREF
  int v61; // [rsp+38h] [rbp-D8h]
  __int64 v62; // [rsp+40h] [rbp-D0h]
  __int64 v63; // [rsp+48h] [rbp-C8h]
  _QWORD v64[2]; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v65; // [rsp+60h] [rbp-B0h]
  __int16 v66; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v67; // [rsp+78h] [rbp-98h]
  __int16 v68; // [rsp+80h] [rbp-90h]
  __int64 v69; // [rsp+88h] [rbp-88h]
  _QWORD v70[4]; // [rsp+90h] [rbp-80h] BYREF
  const char *v71; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v72; // [rsp+B8h] [rbp-58h]
  __int64 (__fastcall *v73)(__int64, __int64, unsigned int); // [rsp+C0h] [rbp-50h]
  __int64 v74; // [rsp+C8h] [rbp-48h]
  __int64 v75; // [rsp+D0h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 24);
  if ( v4 > 239 )
  {
    if ( (unsigned int)(v4 - 242) <= 1 )
    {
      v59 = 1;
      v5 = 1;
    }
    else
    {
      v59 = 0;
      v5 = 0;
      if ( v4 == 258 )
      {
LABEL_5:
        v6 = sub_2FE5E70(**(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
        goto LABEL_6;
      }
    }
  }
  else
  {
    if ( v4 > 237 )
    {
      v59 = 1;
      v5 = 1;
    }
    else
    {
      v59 = (unsigned int)(v4 - 101) < 0x30;
      v5 = v59;
    }
    if ( v4 == 109 )
      goto LABEL_5;
  }
  v6 = sub_2FE5EA0(**(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
LABEL_6:
  v7 = a1[1];
  if ( !*(_QWORD *)(*a1 + 8LL * v6 + 525288) )
  {
    v46 = *(_QWORD *)(v7 + 64);
    v71 = "Don't know how to soften fpowi to fpow";
    LOWORD(v75) = 259;
    sub_B6ECE0(v46, (__int64)&v71);
    v47 = (_QWORD *)a1[1];
    v48 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
    v49 = **(unsigned __int16 **)(a2 + 48);
    v71 = 0;
    LODWORD(v72) = 0;
    v17 = sub_33F17F0(v47, 51, (__int64)&v71, v49, v48);
    if ( v71 )
      sub_B91220((__int64)&v71, (__int64)v71);
    return v17;
  }
  v56 = *(unsigned int *)(**(_QWORD **)(v7 + 24) + 172LL);
  v54 = 5LL * (unsigned int)(v5 + 1);
  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + v54 * 8) + 48LL)
     + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + v54 * 8 + 8);
  v9 = *(_WORD *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  LOWORD(v70[0]) = v9;
  v70[1] = v10;
  if ( v9 )
  {
    if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
      BUG();
    v12 = 16LL * (v9 - 1);
    v11 = *(_QWORD *)&byte_444C4A0[v12];
    LOBYTE(v12) = byte_444C4A0[v12 + 8];
  }
  else
  {
    v11 = sub_3007260((__int64)v70);
    v62 = v11;
    v63 = v12;
  }
  v71 = (const char *)v11;
  LOBYTE(v72) = v12;
  if ( v56 != sub_CA1930(&v71) )
  {
    v13 = *(_QWORD *)(a1[1] + 64);
    v71 = "POWI exponent does not match sizeof(int)";
    LOWORD(v75) = 259;
    sub_B6ECE0(v13, (__int64)&v71);
    v14 = (_QWORD *)a1[1];
    v15 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
    v16 = **(unsigned __int16 **)(a2 + 48);
    v71 = 0;
    LODWORD(v72) = 0;
    v17 = sub_33F17F0(v14, 51, (__int64)&v71, v16, v15);
    if ( v71 )
      sub_B91220((__int64)&v71, (__int64)v71);
    return v17;
  }
  v19 = a1[1];
  v20 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v21 = *(__int16 **)(a2 + 48);
  v22 = *v21;
  v23 = *((_QWORD *)v21 + 1);
  if ( v20 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v71, *a1, *(_QWORD *)(v19 + 64), v22, v23);
    v24 = v73;
    v25 = (unsigned __int16)v72;
  }
  else
  {
    v25 = v20(*a1, *(_QWORD *)(v19 + 64), v22, v23);
    v24 = v50;
  }
  v52 = v24;
  v26 = 5 * v5;
  v57 = v25;
  v27 = sub_3805E70(
          (__int64)a1,
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + v26 * 8),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + v26 * 8 + 8));
  v28 = v57;
  v29 = v52;
  v64[0] = v27;
  v30 = *(__int64 **)(a2 + 40);
  v64[1] = v31;
  v32 = (unsigned int *)&v30[v54];
  v65 = _mm_loadu_si128((const __m128i *)&v30[v54]);
  if ( v59 )
  {
    v33 = *v30;
    v34 = v30[1];
  }
  else
  {
    v33 = 0;
    v34 = 0;
  }
  LOBYTE(v75) = 4;
  v35 = (_WORD *)*a1;
  v36 = *(_QWORD *)(v30[v26] + 48) + 16LL * LODWORD(v30[v26 + 1]);
  v37 = *(_WORD *)v36;
  v38 = *(_QWORD *)(v36 + 8);
  v66 = v37;
  v67 = v38;
  v39 = *(_QWORD *)(*(_QWORD *)v32 + 48LL) + 16LL * v32[2];
  v40 = *(_WORD *)v39;
  v69 = *(_QWORD *)(v39 + 8);
  v41 = *(__int16 **)(a2 + 48);
  v68 = v40;
  v42 = *v41;
  v43 = *((_QWORD *)v41 + 1);
  v71 = (const char *)&v66;
  v44 = *(_QWORD *)(a2 + 80);
  v72 = 2;
  LOWORD(v73) = v42;
  v74 = v43;
  LOBYTE(v75) = 20;
  v60 = v44;
  if ( v44 )
  {
    v51 = v52;
    v53 = v57;
    v55 = v33;
    v58 = v35;
    sub_B96E90((__int64)&v60, v44, 1);
    v29 = v51;
    v28 = v53;
    v33 = v55;
    v35 = v58;
  }
  v45 = a1[1];
  v61 = *(_DWORD *)(a2 + 72);
  sub_3494590(
    (__int64)v70,
    v35,
    v45,
    v6,
    v28,
    v29,
    (__int64)v64,
    2u,
    (__int64)v71,
    v72,
    (unsigned int)v73,
    v74,
    v75,
    (__int64)&v60,
    v33,
    v34);
  if ( v60 )
    sub_B91220((__int64)&v60, v60);
  if ( v59 )
    sub_3760E70((__int64)a1, a2, 1, v70[2], v70[3]);
  return (_QWORD *)v70[0];
}
