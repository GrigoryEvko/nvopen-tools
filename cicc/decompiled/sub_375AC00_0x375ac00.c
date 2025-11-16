// Function: sub_375AC00
// Address: 0x375ac00
//
__m128i *__fastcall sub_375AC00(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        unsigned int a4,
        unsigned __int64 a5)
{
  __int64 v8; // rsi
  __int64 v9; // rdi
  unsigned int v10; // ebx
  __int64 v11; // r15
  unsigned int v12; // eax
  __int64 v13; // r8
  __int64 v14; // r15
  unsigned __int16 v15; // ax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdx
  char v20; // al
  _QWORD *v21; // rax
  _QWORD *v22; // rdi
  unsigned __int64 v23; // rdx
  __m128i *v24; // rax
  __int16 v25; // cx
  __int64 *v26; // rdi
  __int64 v27; // rdx
  __m128i *v28; // r12
  __int64 v30; // [rsp+0h] [rbp-D0h]
  __int64 v31; // [rsp+0h] [rbp-D0h]
  __int64 v32; // [rsp+8h] [rbp-C8h]
  __int64 v35; // [rsp+30h] [rbp-A0h] BYREF
  int v36; // [rsp+38h] [rbp-98h]
  __int64 v37; // [rsp+48h] [rbp-88h]
  __int64 v38; // [rsp+50h] [rbp-80h]
  __int64 v39; // [rsp+58h] [rbp-78h]
  __int64 v40; // [rsp+60h] [rbp-70h]
  __int64 v41; // [rsp+68h] [rbp-68h]
  __int64 v42; // [rsp+70h] [rbp-60h]
  __int64 v43; // [rsp+80h] [rbp-50h] BYREF
  __int64 v44; // [rsp+88h] [rbp-48h]
  __int64 v45; // [rsp+90h] [rbp-40h]
  __int64 v46; // [rsp+98h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  v35 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v35, v8, 1);
  v9 = *(_QWORD *)(a1 + 8);
  v36 = *(_DWORD *)(a2 + 72);
  v10 = sub_33CD850(v9, a4, a5, 0);
  v11 = 16LL * (unsigned int)a3;
  v12 = sub_33CD850(
          *(_QWORD *)(a1 + 8),
          *(unsigned __int16 *)(v11 + *(_QWORD *)(a2 + 48)),
          *(_QWORD *)(v11 + *(_QWORD *)(a2 + 48) + 8),
          0);
  v13 = *(_QWORD *)(a1 + 8);
  if ( (unsigned __int8)v10 < (unsigned __int8)v12 )
    v10 = v12;
  v14 = *(_QWORD *)(a2 + 48) + v11;
  v15 = *(_WORD *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  LOWORD(v43) = v15;
  v44 = v16;
  if ( v15 )
  {
    if ( v15 == 1 || (unsigned __int16)(v15 - 504) <= 7u )
      BUG();
    v18 = *(_QWORD *)&byte_444C4A0[16 * v15 - 16];
    v20 = byte_444C4A0[16 * v15 - 8];
  }
  else
  {
    v30 = v13;
    v17 = sub_3007260((__int64)&v43);
    v13 = v30;
    v38 = v17;
    v18 = v17;
    v39 = v19;
    v20 = v19;
  }
  LOBYTE(v37) = v20;
  v21 = sub_33EDE90(v13, (unsigned __int64)(v18 + 7) >> 3, v37, v10);
  v22 = *(_QWORD **)(a1 + 8);
  v41 = 0;
  LODWORD(v42) = 0;
  BYTE4(v42) = 0;
  v32 = v23;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v40 = 0;
  v31 = (__int64)v21;
  v24 = sub_33F4560(
          v22,
          (unsigned __int64)(v22 + 36),
          0,
          (__int64)&v35,
          a2,
          a3,
          (unsigned __int64)v21,
          v23,
          0,
          v42,
          v10,
          0,
          (__int64)&v43);
  HIBYTE(v25) = 1;
  v26 = *(__int64 **)(a1 + 8);
  LOBYTE(v25) = v10;
  BYTE4(v42) = 0;
  v41 = 0;
  LODWORD(v42) = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v40 = 0;
  v28 = sub_33F1F00(v26, a4, a5, (__int64)&v35, (__int64)v24, v27, v31, v32, 0, v42, v25, 0, (__int64)&v43, 0);
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  return v28;
}
