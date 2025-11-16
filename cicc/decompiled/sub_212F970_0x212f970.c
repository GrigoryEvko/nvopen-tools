// Function: sub_212F970
// Address: 0x212f970
//
unsigned __int64 __fastcall sub_212F970(
        __int64 **a1,
        __int64 a2,
        _DWORD *a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  __int64 v8; // rsi
  __int64 v9; // rax
  const void **v10; // r8
  __int128 v11; // rax
  __int64 *v12; // r15
  __int64 (__fastcall *v13)(__int64 *, __int64, __int64, unsigned __int64, const void **); // r14
  __int64 v14; // rax
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  unsigned __int64 v17; // r14
  __int16 *v18; // r15
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // edx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // r15
  __int64 v27; // r14
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int128 v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // r10
  __int64 v33; // r11
  unsigned __int64 v34; // rcx
  const void **v35; // r8
  unsigned __int64 v36; // r14
  __int64 v37; // r13
  __int16 *v38; // r15
  char v39; // al
  unsigned int v40; // esi
  __int64 *v41; // rax
  const void **v42; // r8
  int v43; // edx
  __int64 v44; // rax
  __int64 v45; // rsi
  int v46; // edx
  unsigned __int64 result; // rax
  bool v48; // al
  unsigned __int64 v49; // [rsp-10h] [rbp-E0h]
  __int64 *v50; // [rsp+0h] [rbp-D0h]
  const void **v51; // [rsp+0h] [rbp-D0h]
  __int64 v52; // [rsp+8h] [rbp-C8h]
  const void **v53; // [rsp+8h] [rbp-C8h]
  unsigned int v54; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v55; // [rsp+8h] [rbp-C8h]
  const void **v56; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v57; // [rsp+10h] [rbp-C0h]
  __int64 *v58; // [rsp+10h] [rbp-C0h]
  __int64 *v59; // [rsp+10h] [rbp-C0h]
  __int64 v60; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v61; // [rsp+20h] [rbp-B0h]
  __int128 v62; // [rsp+20h] [rbp-B0h]
  __int128 v63; // [rsp+30h] [rbp-A0h]
  __int64 *v64; // [rsp+30h] [rbp-A0h]
  __int64 v67; // [rsp+70h] [rbp-60h] BYREF
  int v68; // [rsp+78h] [rbp-58h]
  unsigned __int64 v69; // [rsp+80h] [rbp-50h] BYREF
  const void **v70; // [rsp+88h] [rbp-48h]
  char v71[8]; // [rsp+90h] [rbp-40h] BYREF
  __int64 v72; // [rsp+98h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 72);
  v67 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v67, v8, 2);
  v68 = *(_DWORD *)(a2 + 64);
  sub_20174B0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, (_DWORD *)a4);
  v9 = *(_QWORD *)(*(_QWORD *)a3 + 40LL) + 16LL * (unsigned int)a3[2];
  v10 = *(const void ***)(v9 + 8);
  v50 = a1[1];
  LOBYTE(v69) = *(_BYTE *)v9;
  v70 = v10;
  *(_QWORD *)&v11 = sub_1D38BB0((__int64)v50, 0, (__int64)&v67, (unsigned int)v69, v10, 0, a5, a6, a7, 0);
  v12 = *a1;
  v63 = v11;
  *(_QWORD *)&v11 = a1[1];
  v56 = v70;
  v13 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, unsigned __int64, const void **))(**a1 + 264);
  v61 = v69;
  v52 = *(_QWORD *)(v11 + 48);
  v14 = sub_1E0A0C0(*(_QWORD *)(v11 + 32));
  v15 = v13(v12, v14, v52, v61, v56);
  v53 = (const void **)v16;
  v57 = v15;
  v17 = *(_QWORD *)a4;
  v18 = *(__int16 **)(a4 + 8);
  v21 = sub_1D28D50(v50, 0x16u, v16, a4, v19, v20);
  v58 = sub_1D3A900(v50, 0x89u, (__int64)&v67, v57, v53, 0, (__m128)a5, a6, a7, v17, v18, v63, v21, v22);
  v54 = v23;
  v24 = sub_1D309E0(
          a1[1],
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v67,
          (unsigned int)v69,
          v70,
          0,
          *(double *)a5.m128i_i64,
          a6,
          *(double *)a7.m128i_i64,
          *(_OWORD *)a3);
  v26 = v25;
  v27 = v24;
  *(_QWORD *)&v62 = sub_1D309E0(
                      a1[1],
                      133,
                      (__int64)&v67,
                      (unsigned int)v69,
                      v70,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      *(double *)a7.m128i_i64,
                      *(_OWORD *)a4);
  *((_QWORD *)&v62 + 1) = v28;
  v64 = a1[1];
  if ( (_BYTE)v69 )
    v29 = sub_2127930(v69);
  else
    v29 = sub_1F58D40((__int64)&v69);
  *(_QWORD *)&v30 = sub_1D38BB0((__int64)v64, v29, (__int64)&v67, (unsigned int)v69, v70, 0, a5, a6, a7, 0);
  v32 = sub_1D332F0(v64, 52, (__int64)&v67, (unsigned int)v69, v70, 0, *(double *)a5.m128i_i64, a6, a7, v27, v26, v30);
  v33 = v31;
  v34 = v69;
  v35 = v70;
  v36 = (unsigned __int64)v58;
  v37 = v58[5] + 16LL * v54;
  v38 = (__int16 *)v54;
  v39 = *(_BYTE *)v37;
  v72 = *(_QWORD *)(v37 + 8);
  v71[0] = v39;
  if ( v39 )
  {
    v40 = ((unsigned __int8)(v39 - 14) < 0x60u) + 134;
  }
  else
  {
    v51 = v70;
    v55 = v69;
    v59 = v32;
    v60 = v31;
    v48 = sub_1F58D20((__int64)v71);
    v35 = v51;
    v34 = v55;
    v32 = v59;
    v33 = v60;
    v40 = 134 - (!v48 - 1);
  }
  v41 = sub_1D3A900(v64, v40, (__int64)&v67, v34, v35, 0, (__m128)a5, a6, a7, v36, v38, v62, (__int64)v32, v33);
  v42 = v70;
  *(_QWORD *)a3 = v41;
  a3[2] = v43;
  v44 = sub_1D38BB0((__int64)a1[1], 0, (__int64)&v67, (unsigned int)v69, v42, 0, a5, a6, a7, 0);
  v45 = v67;
  *(_QWORD *)a4 = v44;
  *(_DWORD *)(a4 + 8) = v46;
  result = v49;
  if ( v45 )
    return sub_161E7C0((__int64)&v67, v45);
  return result;
}
