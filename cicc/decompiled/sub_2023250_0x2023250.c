// Function: sub_2023250
// Address: 0x2023250
//
unsigned __int64 __fastcall sub_2023250(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        __m128i a7)
{
  unsigned __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 *v18; // rdi
  unsigned int v19; // esi
  const void ***v20; // rax
  int v21; // edx
  unsigned __int64 v22; // rdx
  const void ***v23; // rax
  __int64 *v24; // rax
  __int64 v25; // rsi
  unsigned int v26; // edx
  unsigned __int64 result; // rax
  int v28; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v29; // [rsp+20h] [rbp-A0h] BYREF
  __int16 *v30; // [rsp+28h] [rbp-98h]
  unsigned __int64 v31; // [rsp+30h] [rbp-90h] BYREF
  __int16 *v32; // [rsp+38h] [rbp-88h]
  __int128 v33; // [rsp+40h] [rbp-80h] BYREF
  __int128 v34; // [rsp+50h] [rbp-70h] BYREF
  __int64 v35; // [rsp+60h] [rbp-60h] BYREF
  __int64 v36; // [rsp+68h] [rbp-58h]
  __int64 v37; // [rsp+70h] [rbp-50h] BYREF
  __int64 v38; // [rsp+78h] [rbp-48h]
  __int64 v39; // [rsp+80h] [rbp-40h] BYREF
  int v40; // [rsp+88h] [rbp-38h]

  v11 = *(unsigned __int64 **)(a2 + 32);
  LODWORD(v30) = 0;
  LODWORD(v32) = 0;
  v29 = 0;
  v12 = v11[1];
  v31 = 0;
  sub_2017DE0(a1, *v11, v12, &v29, &v31);
  v13 = *(_QWORD *)(a2 + 32);
  DWORD2(v33) = 0;
  DWORD2(v34) = 0;
  v14 = *(_QWORD *)(v13 + 48);
  *(_QWORD *)&v33 = 0;
  *(_QWORD *)&v34 = 0;
  sub_2017DE0(a1, *(_QWORD *)(v13 + 40), v14, &v33, &v34);
  v15 = *(_QWORD *)(a2 + 32);
  LODWORD(v36) = 0;
  LODWORD(v38) = 0;
  v16 = *(_QWORD *)(v15 + 88);
  v35 = 0;
  v37 = 0;
  sub_2017DE0(a1, *(_QWORD *)(v15 + 80), v16, &v35, &v37);
  v17 = *(_QWORD *)(a2 + 72);
  v39 = v17;
  if ( v17 )
    sub_1623A60((__int64)&v39, v17, 2);
  v18 = *(__int64 **)(a1 + 8);
  v19 = *(unsigned __int16 *)(a2 + 24);
  v40 = *(_DWORD *)(a2 + 64);
  v20 = (const void ***)(*(_QWORD *)(v29 + 40) + 16LL * (unsigned int)v30);
  *(_QWORD *)a3 = sub_1D3A900(
                    v18,
                    v19,
                    (__int64)&v39,
                    *(unsigned __int8 *)v20,
                    v20[1],
                    0,
                    a5,
                    a6,
                    a7,
                    v29,
                    v30,
                    v33,
                    v35,
                    v36);
  v28 = v21;
  v22 = v31;
  *(_DWORD *)(a3 + 8) = v28;
  v23 = (const void ***)(*(_QWORD *)(v22 + 40) + 16LL * (unsigned int)v32);
  v24 = sub_1D3A900(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v39,
          *(unsigned __int8 *)v23,
          v23[1],
          0,
          a5,
          a6,
          a7,
          v31,
          v32,
          v34,
          v37,
          v38);
  v25 = v39;
  *(_QWORD *)a4 = v24;
  result = v26;
  *(_DWORD *)(a4 + 8) = v26;
  if ( v25 )
    return sub_161E7C0((__int64)&v39, v25);
  return result;
}
