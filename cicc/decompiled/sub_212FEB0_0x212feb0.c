// Function: sub_212FEB0
// Address: 0x212feb0
//
unsigned __int64 __fastcall sub_212FEB0(
        __int64 **a1,
        __int64 a2,
        __int16 **a3,
        _DWORD *a4,
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
  __int64 v24; // rdx
  __int64 v25; // r14
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // r15
  unsigned int v28; // eax
  __int128 v29; // rax
  __int64 v30; // rdx
  __int64 *v31; // r10
  __int64 v32; // r11
  unsigned __int64 v33; // rcx
  const void **v34; // r8
  unsigned __int64 v35; // r14
  __int64 v36; // r13
  __int16 *v37; // r15
  char v38; // al
  unsigned int v39; // esi
  __int64 *v40; // rax
  const void **v41; // r8
  int v42; // edx
  __int64 v43; // rax
  __int64 v44; // rsi
  int v45; // edx
  unsigned __int64 result; // rax
  bool v47; // al
  unsigned __int64 v48; // [rsp-10h] [rbp-E0h]
  __int64 *v49; // [rsp+0h] [rbp-D0h]
  const void **v50; // [rsp+0h] [rbp-D0h]
  __int64 v51; // [rsp+8h] [rbp-C8h]
  const void **v52; // [rsp+8h] [rbp-C8h]
  unsigned int v53; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v54; // [rsp+8h] [rbp-C8h]
  const void **v55; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v56; // [rsp+10h] [rbp-C0h]
  __int64 *v57; // [rsp+10h] [rbp-C0h]
  __int64 *v58; // [rsp+10h] [rbp-C0h]
  __int64 v59; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v60; // [rsp+20h] [rbp-B0h]
  __int128 v61; // [rsp+20h] [rbp-B0h]
  __int128 v62; // [rsp+30h] [rbp-A0h]
  __int64 *v63; // [rsp+30h] [rbp-A0h]
  __int64 v66; // [rsp+70h] [rbp-60h] BYREF
  int v67; // [rsp+78h] [rbp-58h]
  unsigned __int64 v68; // [rsp+80h] [rbp-50h] BYREF
  const void **v69; // [rsp+88h] [rbp-48h]
  char v70[8]; // [rsp+90h] [rbp-40h] BYREF
  __int64 v71; // [rsp+98h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 72);
  v66 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v66, v8, 2);
  v67 = *(_DWORD *)(a2 + 64);
  sub_20174B0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, a4);
  v9 = *((_QWORD *)*a3 + 5) + 16LL * *((unsigned int *)a3 + 2);
  v10 = *(const void ***)(v9 + 8);
  v49 = a1[1];
  LOBYTE(v68) = *(_BYTE *)v9;
  v69 = v10;
  *(_QWORD *)&v11 = sub_1D38BB0((__int64)v49, 0, (__int64)&v66, (unsigned int)v68, v10, 0, a5, a6, a7, 0);
  v12 = *a1;
  v62 = v11;
  *(_QWORD *)&v11 = a1[1];
  v55 = v69;
  v13 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, unsigned __int64, const void **))(**a1 + 264);
  v60 = v68;
  v51 = *(_QWORD *)(v11 + 48);
  v14 = sub_1E0A0C0(*(_QWORD *)(v11 + 32));
  v15 = v13(v12, v14, v51, v60, v55);
  v52 = (const void **)v16;
  v56 = v15;
  v17 = (unsigned __int64)*a3;
  v18 = a3[1];
  v21 = sub_1D28D50(v49, 0x16u, v16, (__int64)a3, v19, v20);
  v57 = sub_1D3A900(v49, 0x89u, (__int64)&v66, v56, v52, 0, (__m128)a5, a6, a7, v17, v18, v62, v21, v22);
  v53 = v23;
  *(_QWORD *)&v61 = sub_1D309E0(
                      a1[1],
                      132,
                      (__int64)&v66,
                      (unsigned int)v68,
                      v69,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      *(double *)a7.m128i_i64,
                      *(_OWORD *)a3);
  *((_QWORD *)&v61 + 1) = v24;
  v25 = sub_1D309E0(
          a1[1],
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v66,
          (unsigned int)v68,
          v69,
          0,
          *(double *)a5.m128i_i64,
          a6,
          *(double *)a7.m128i_i64,
          *(_OWORD *)a4);
  v27 = v26;
  v63 = a1[1];
  if ( (_BYTE)v68 )
    v28 = sub_2127930(v68);
  else
    v28 = sub_1F58D40((__int64)&v68);
  *(_QWORD *)&v29 = sub_1D38BB0((__int64)v63, v28, (__int64)&v66, (unsigned int)v68, v69, 0, a5, a6, a7, 0);
  v31 = sub_1D332F0(v63, 52, (__int64)&v66, (unsigned int)v68, v69, 0, *(double *)a5.m128i_i64, a6, a7, v25, v27, v29);
  v32 = v30;
  v33 = v68;
  v34 = v69;
  v35 = (unsigned __int64)v57;
  v36 = v57[5] + 16LL * v53;
  v37 = (__int16 *)v53;
  v38 = *(_BYTE *)v36;
  v71 = *(_QWORD *)(v36 + 8);
  v70[0] = v38;
  if ( v38 )
  {
    v39 = ((unsigned __int8)(v38 - 14) < 0x60u) + 134;
  }
  else
  {
    v50 = v69;
    v54 = v68;
    v58 = v31;
    v59 = v30;
    v47 = sub_1F58D20((__int64)v70);
    v34 = v50;
    v33 = v54;
    v31 = v58;
    v32 = v59;
    v39 = 134 - (!v47 - 1);
  }
  v40 = sub_1D3A900(v63, v39, (__int64)&v66, v33, v34, 0, (__m128)a5, a6, a7, v35, v37, v61, (__int64)v31, v32);
  v41 = v69;
  *a3 = (__int16 *)v40;
  *((_DWORD *)a3 + 2) = v42;
  v43 = sub_1D38BB0((__int64)a1[1], 0, (__int64)&v66, (unsigned int)v68, v41, 0, a5, a6, a7, 0);
  v44 = v66;
  *(_QWORD *)a4 = v43;
  a4[2] = v45;
  result = v48;
  if ( v44 )
    return sub_161E7C0((__int64)&v66, v44);
  return result;
}
