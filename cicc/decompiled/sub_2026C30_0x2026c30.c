// Function: sub_2026C30
// Address: 0x2026c30
//
unsigned __int64 __fastcall sub_2026C30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        double a7)
{
  __int64 v9; // rsi
  __int64 v10; // rsi
  char *v11; // rax
  char v12; // dl
  __int64 v13; // rax
  int v14; // edx
  _QWORD *v15; // rdi
  __int64 v16; // r9
  _QWORD *v17; // rbx
  unsigned int v18; // edx
  unsigned int v19; // r13d
  __int64 v20; // rsi
  unsigned __int64 result; // rax
  __int64 v22; // [rsp+8h] [rbp-A8h]
  unsigned __int8 v24; // [rsp+1Fh] [rbp-91h]
  __int64 v25; // [rsp+40h] [rbp-70h] BYREF
  int v26; // [rsp+48h] [rbp-68h]
  _QWORD v27[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v28; // [rsp+60h] [rbp-50h] BYREF
  const void **v29; // [rsp+68h] [rbp-48h]
  unsigned __int8 v30; // [rsp+70h] [rbp-40h]
  __int64 v31; // [rsp+78h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 72);
  v25 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v25, v9, 2);
  v10 = *(_QWORD *)(a1 + 8);
  v26 = *(_DWORD *)(a2 + 64);
  v11 = *(char **)(a2 + 40);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  LOBYTE(v27[0]) = v12;
  v27[1] = v13;
  sub_1D19A30((__int64)&v28, v10, v27);
  v24 = v30;
  v22 = v31;
  *(_QWORD *)a3 = sub_1D309E0(
                    *(__int64 **)(a1 + 8),
                    111,
                    (__int64)&v25,
                    v28,
                    v29,
                    0,
                    a5,
                    a6,
                    a7,
                    *(_OWORD *)*(_QWORD *)(a2 + 32));
  v28 = 0;
  *(_DWORD *)(a3 + 8) = v14;
  v15 = *(_QWORD **)(a1 + 8);
  LODWORD(v29) = 0;
  v17 = sub_1D2B300(v15, 0x30u, (__int64)&v28, v24, v22, v16);
  v19 = v18;
  if ( v28 )
    sub_161E7C0((__int64)&v28, v28);
  v20 = v25;
  *(_QWORD *)a4 = v17;
  result = v19;
  *(_DWORD *)(a4 + 8) = v19;
  if ( v20 )
    return sub_161E7C0((__int64)&v25, v20);
  return result;
}
