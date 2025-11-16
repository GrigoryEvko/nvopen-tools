// Function: sub_20230C0
// Address: 0x20230c0
//
unsigned __int64 __fastcall sub_20230C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        __m128i a7)
{
  unsigned __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rsi
  int v14; // eax
  unsigned int v15; // r15d
  unsigned int v16; // ebx
  const void ***v17; // rax
  __int64 *v18; // rax
  int v19; // edx
  __int64 v20; // rdx
  const void ***v21; // rax
  __int64 *v22; // rax
  __int64 v23; // rsi
  unsigned int v24; // edx
  unsigned __int64 result; // rax
  int v27; // [rsp+28h] [rbp-88h]
  __int64 v28; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v29; // [rsp+38h] [rbp-78h]
  __int64 v30; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v31; // [rsp+48h] [rbp-68h]
  __int128 v32; // [rsp+50h] [rbp-60h] BYREF
  __int128 v33; // [rsp+60h] [rbp-50h] BYREF
  __int64 v34; // [rsp+70h] [rbp-40h] BYREF
  int v35; // [rsp+78h] [rbp-38h]

  v9 = *(unsigned __int64 **)(a2 + 32);
  LODWORD(v29) = 0;
  LODWORD(v31) = 0;
  v28 = 0;
  v10 = v9[1];
  v30 = 0;
  sub_2017DE0(a1, *v9, v10, &v28, &v30);
  v11 = *(_QWORD *)(a2 + 32);
  DWORD2(v32) = 0;
  DWORD2(v33) = 0;
  v12 = *(_QWORD *)(v11 + 48);
  *(_QWORD *)&v32 = 0;
  *(_QWORD *)&v33 = 0;
  sub_2017DE0(a1, *(_QWORD *)(v11 + 40), v12, &v32, &v33);
  v13 = *(_QWORD *)(a2 + 72);
  v34 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v34, v13, 2);
  v14 = *(_DWORD *)(a2 + 64);
  v15 = *(unsigned __int16 *)(a2 + 80);
  v16 = *(unsigned __int16 *)(a2 + 24);
  v35 = v14;
  v17 = (const void ***)(*(_QWORD *)(v28 + 40) + 16LL * (unsigned int)v29);
  v18 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          v16,
          (__int64)&v34,
          *(unsigned __int8 *)v17,
          v17[1],
          v15,
          a5,
          a6,
          a7,
          v28,
          v29,
          v32);
  v27 = v19;
  v20 = v30;
  *(_QWORD *)a3 = v18;
  *(_DWORD *)(a3 + 8) = v27;
  v21 = (const void ***)(*(_QWORD *)(v20 + 40) + 16LL * (unsigned int)v31);
  v22 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          v16,
          (__int64)&v34,
          *(unsigned __int8 *)v21,
          v21[1],
          v15,
          a5,
          a6,
          a7,
          v30,
          v31,
          v33);
  v23 = v34;
  *(_QWORD *)a4 = v22;
  result = v24;
  *(_DWORD *)(a4 + 8) = v24;
  if ( v23 )
    return sub_161E7C0((__int64)&v34, v23);
  return result;
}
