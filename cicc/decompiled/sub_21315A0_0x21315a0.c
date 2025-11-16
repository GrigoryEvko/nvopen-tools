// Function: sub_21315A0
// Address: 0x21315a0
//
unsigned __int64 __fastcall sub_21315A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        __m128i a7)
{
  __int64 v10; // rsi
  int v11; // eax
  unsigned __int64 *v12; // rax
  __int64 v13; // rdx
  const void ***v14; // rax
  int v15; // edx
  __int64 v16; // rdx
  const void ***v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rsi
  unsigned int v20; // edx
  unsigned __int64 result; // rax
  int v22; // [rsp+18h] [rbp-88h]
  __int64 v23; // [rsp+20h] [rbp-80h] BYREF
  int v24; // [rsp+28h] [rbp-78h]
  __int64 v25; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v26; // [rsp+38h] [rbp-68h]
  __int64 v27; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v28; // [rsp+48h] [rbp-58h]
  __int128 v29; // [rsp+50h] [rbp-50h] BYREF
  __int128 v30; // [rsp+60h] [rbp-40h] BYREF

  v10 = *(_QWORD *)(a2 + 72);
  v23 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v23, v10, 2);
  v11 = *(_DWORD *)(a2 + 64);
  LODWORD(v26) = 0;
  LODWORD(v28) = 0;
  v24 = v11;
  v12 = *(unsigned __int64 **)(a2 + 32);
  DWORD2(v29) = 0;
  DWORD2(v30) = 0;
  v13 = v12[1];
  v25 = 0;
  v27 = 0;
  *(_QWORD *)&v29 = 0;
  *(_QWORD *)&v30 = 0;
  sub_20174B0(a1, *v12, v13, &v25, &v27);
  sub_20174B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL), &v29, &v30);
  v14 = (const void ***)(*(_QWORD *)(v25 + 40) + 16LL * (unsigned int)v26);
  *(_QWORD *)a3 = sub_1D332F0(
                    *(__int64 **)(a1 + 8),
                    *(unsigned __int16 *)(a2 + 24),
                    (__int64)&v23,
                    *(unsigned __int8 *)v14,
                    v14[1],
                    0,
                    a5,
                    a6,
                    a7,
                    v25,
                    v26,
                    v29);
  v22 = v15;
  v16 = v25;
  *(_DWORD *)(a3 + 8) = v22;
  v17 = (const void ***)(*(_QWORD *)(v16 + 40) + 16LL * (unsigned int)v26);
  v18 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v23,
          *(unsigned __int8 *)v17,
          v17[1],
          0,
          a5,
          a6,
          a7,
          v27,
          v28,
          v30);
  v19 = v23;
  *(_QWORD *)a4 = v18;
  result = v20;
  *(_DWORD *)(a4 + 8) = v20;
  if ( v19 )
    return sub_161E7C0((__int64)&v23, v19);
  return result;
}
