// Function: sub_B34600
// Address: 0xb34600
//
__int64 __fastcall sub_B34600(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        unsigned int a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 *v19; // rsi
  __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rdx
  __int64 *v26; // rsi
  __int64 v27; // rdx
  __int64 *v28; // rsi
  __int64 v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rdx
  __int64 *v35; // rax
  unsigned int v38; // [rsp+18h] [rbp-A8h]
  _QWORD v39[4]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v40[4]; // [rsp+40h] [rbp-80h] BYREF
  _DWORD v41[8]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v42; // [rsp+80h] [rbp-40h]

  v15 = *(_QWORD *)(a1 + 72);
  v40[1] = a4;
  v40[2] = a6;
  v40[0] = a2;
  v16 = sub_BCB2D0(v15);
  v40[3] = sub_ACD640(v16, a7, 0);
  v39[0] = *(_QWORD *)(a2 + 8);
  v39[1] = *(_QWORD *)(a4 + 8);
  v39[2] = *(_QWORD *)(a6 + 8);
  v42 = 257;
  v17 = sub_B33D10(a1, 0xEFu, (__int64)v39, 3, (int)v40, 4, v38, (__int64)v41);
  v19 = (__int64 *)sub_BD5C60(v17, 239, v18);
  *(_QWORD *)(v17 + 72) = sub_A7B980((__int64 *)(v17 + 72), v19, 1, 86);
  v21 = (__int64 *)sub_BD5C60(v17, v19, v20);
  v22 = a3;
  v23 = sub_A77A40(v21, a3);
  v41[0] = 0;
  v24 = v23;
  v26 = (__int64 *)sub_BD5C60(v17, v22, v25);
  *(_QWORD *)(v17 + 72) = sub_A7B660((__int64 *)(v17 + 72), v26, v41, 1, v24);
  v28 = (__int64 *)sub_BD5C60(v17, v26, v27);
  *(_QWORD *)(v17 + 72) = sub_A7B980((__int64 *)(v17 + 72), v28, 2, 86);
  v30 = (__int64 *)sub_BD5C60(v17, v28, v29);
  v31 = a5;
  v32 = sub_A77A40(v30, a5);
  v41[0] = 1;
  v33 = v32;
  v35 = (__int64 *)sub_BD5C60(v17, v31, v34);
  *(_QWORD *)(v17 + 72) = sub_A7B660((__int64 *)(v17 + 72), v35, v41, 1, v33);
  if ( a8 )
    sub_B99FD0(v17, 1, a8);
  if ( a9 )
    sub_B99FD0(v17, 5, a9);
  if ( a10 )
    sub_B99FD0(v17, 7, a10);
  if ( a11 )
    sub_B99FD0(v17, 8, a11);
  return v17;
}
