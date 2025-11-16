// Function: sub_21779A0
// Address: 0x21779a0
//
__int64 *__fastcall sub_21779A0(__int64 a1, double a2, double a3, __m128i a4, __int64 a5, __int64 *a6, char a7)
{
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r9
  int v12; // r8d
  __int64 v13; // r10
  __int64 v14; // rcx
  __int64 v15; // r15
  __int64 v16; // r14
  int v17; // eax
  unsigned __int8 v18; // bl
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r9
  __int64 *v24; // r14
  __int128 v26; // [rsp-10h] [rbp-C0h]
  __int64 v27; // [rsp+0h] [rbp-B0h]
  __int64 v28; // [rsp+8h] [rbp-A8h]
  char v29; // [rsp+10h] [rbp-A0h]
  int v30; // [rsp+14h] [rbp-9Ch]
  __int64 v31; // [rsp+18h] [rbp-98h]
  __int64 v32; // [rsp+20h] [rbp-90h] BYREF
  int v33; // [rsp+28h] [rbp-88h]
  unsigned __int8 v34[8]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v35; // [rsp+38h] [rbp-78h]
  char v36; // [rsp+40h] [rbp-70h]
  __int64 v37; // [rsp+48h] [rbp-68h]
  __int64 v38; // [rsp+50h] [rbp-60h] BYREF
  int v39; // [rsp+58h] [rbp-58h]
  __int64 v40; // [rsp+60h] [rbp-50h]
  __int64 v41; // [rsp+68h] [rbp-48h]
  __int64 v42; // [rsp+70h] [rbp-40h]
  __int64 v43; // [rsp+78h] [rbp-38h]

  v9 = *(__int64 **)(a1 + 32);
  v10 = *(_QWORD *)(a1 + 72);
  v11 = *v9;
  v12 = *((_DWORD *)v9 + 2);
  v32 = v10;
  v13 = v9[5];
  v14 = v9[6];
  v15 = v9[10];
  v16 = v9[11];
  if ( v10 )
  {
    v27 = v9[6];
    v28 = v9[5];
    v29 = a7;
    v30 = v12;
    v31 = v11;
    sub_1623A60((__int64)&v32, v10, 2);
    v14 = v27;
    v13 = v28;
    a7 = v29;
    v12 = v30;
    v11 = v31;
  }
  v17 = *(_DWORD *)(a1 + 64);
  v39 = v12;
  v18 = 5 - ((a7 == 0) - 1);
  v38 = v11;
  v33 = v17;
  v19 = sub_1D323C0(a6, v13, v14, (__int64)&v32, v18 & 0xF, 0, a2, a3, *(double *)a4.m128i_i64);
  v41 = v20;
  v40 = v19;
  v21 = sub_1D323C0(a6, v15, v16, (__int64)&v32, 5, 0, a2, a3, *(double *)a4.m128i_i64);
  v34[0] = v18;
  v42 = v21;
  v43 = v22;
  *((_QWORD *)&v26 + 1) = 3;
  *(_QWORD *)&v26 = &v38;
  v35 = 0;
  v36 = 1;
  v37 = 0;
  v24 = sub_1D373B0(a6, 0x12Fu, (__int64)&v32, v34, 2, a2, a3, a4, v23, v26);
  if ( v32 )
    sub_161E7C0((__int64)&v32, v32);
  return v24;
}
