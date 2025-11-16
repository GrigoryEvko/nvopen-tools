// Function: sub_20251A0
// Address: 0x20251a0
//
unsigned __int64 __fastcall sub_20251A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        __m128i a7)
{
  unsigned __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  __int128 v18; // rax
  int v19; // edx
  __int64 *v20; // r13
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdi
  const void ***v27; // rdx
  __int64 *v28; // rax
  __int64 v29; // rsi
  unsigned int v30; // edx
  unsigned __int64 result; // rax
  __int128 v32; // [rsp-10h] [rbp-E0h]
  __int64 *v33; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v34; // [rsp+10h] [rbp-C0h]
  __int64 v35; // [rsp+18h] [rbp-B8h]
  __int64 v36; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v37; // [rsp+48h] [rbp-88h]
  __int64 v38; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int64 v39; // [rsp+58h] [rbp-78h]
  __int64 v40; // [rsp+60h] [rbp-70h] BYREF
  int v41; // [rsp+68h] [rbp-68h]
  _QWORD v42[2]; // [rsp+70h] [rbp-60h] BYREF
  __int64 v43[3]; // [rsp+80h] [rbp-50h] BYREF
  unsigned __int64 v44; // [rsp+98h] [rbp-38h]

  v10 = *(unsigned __int64 **)(a2 + 32);
  LODWORD(v37) = 0;
  LODWORD(v39) = 0;
  v36 = 0;
  v11 = v10[1];
  v38 = 0;
  sub_2017DE0(a1, *v10, v11, &v36, &v38);
  v12 = *(_QWORD *)(a2 + 72);
  v40 = v12;
  if ( v12 )
    sub_1623A60((__int64)&v40, v12, 2);
  v13 = *(_QWORD *)(a1 + 8);
  v41 = *(_DWORD *)(a2 + 64);
  v14 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
  v15 = *(_QWORD *)(v14 + 96);
  LOBYTE(v14) = *(_BYTE *)(v14 + 88);
  v42[1] = v15;
  LOBYTE(v42[0]) = v14;
  sub_1D19A30((__int64)v43, v13, v42);
  v34 = v44;
  v33 = *(__int64 **)(a1 + 8);
  v35 = v43[2];
  *(_QWORD *)&v18 = sub_1D2EF30(v33, v43[0], v43[1], v44, v16, v17);
  *(_QWORD *)a3 = sub_1D332F0(
                    v33,
                    *(unsigned __int16 *)(a2 + 24),
                    (__int64)&v40,
                    *(unsigned __int8 *)(*(_QWORD *)(v36 + 40) + 16LL * (unsigned int)v37),
                    *(const void ***)(*(_QWORD *)(v36 + 40) + 16LL * (unsigned int)v37 + 8),
                    0,
                    a5,
                    a6,
                    a7,
                    v36,
                    v37,
                    v18);
  *(_DWORD *)(a3 + 8) = v19;
  v20 = *(__int64 **)(a1 + 8);
  v24 = sub_1D2EF30(v20, v35, v34, v21, v22, v23);
  v26 = v25;
  v27 = (const void ***)(*(_QWORD *)(v38 + 40) + 16LL * (unsigned int)v39);
  *((_QWORD *)&v32 + 1) = v26;
  *(_QWORD *)&v32 = v24;
  v28 = sub_1D332F0(
          v20,
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v40,
          *(unsigned __int8 *)v27,
          v27[1],
          0,
          a5,
          a6,
          a7,
          v38,
          v39,
          v32);
  v29 = v40;
  *(_QWORD *)a4 = v28;
  result = v30;
  *(_DWORD *)(a4 + 8) = v30;
  if ( v29 )
    return sub_161E7C0((__int64)&v40, v29);
  return result;
}
