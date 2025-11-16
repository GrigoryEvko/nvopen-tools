// Function: sub_B34240
// Address: 0xb34240
//
__int64 __fastcall sub_B34240(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        unsigned __int8 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 *v18; // rsi
  __int64 v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // [rsp-10h] [rbp-B0h]
  unsigned int v27; // [rsp+8h] [rbp-98h]
  _QWORD v28[2]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v29[4]; // [rsp+20h] [rbp-80h] BYREF
  _DWORD v30[8]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v31; // [rsp+60h] [rbp-40h]

  v13 = *(_QWORD *)(a1 + 72);
  v29[2] = a4;
  v29[0] = a2;
  v29[1] = a3;
  v14 = sub_BCB2A0(v13);
  v29[3] = sub_ACD640(v14, a6, 0);
  v28[0] = *(_QWORD *)(a2 + 8);
  v28[1] = *(_QWORD *)(a4 + 8);
  v31 = 257;
  v15 = sub_B33D10(a1, 0xF3u, (__int64)v28, 2, (int)v29, 4, v27, (__int64)v30);
  v16 = v15;
  if ( BYTE1(a5) )
  {
    v18 = (__int64 *)sub_BD5C60(v15, 243, v26);
    *(_QWORD *)(v16 + 72) = sub_A7B980((__int64 *)(v16 + 72), v18, 1, 86);
    v20 = (__int64 *)sub_BD5C60(v16, v18, v19);
    v21 = a5;
    v22 = sub_A77A40(v20, a5);
    v30[0] = 0;
    v23 = v22;
    v25 = (__int64 *)sub_BD5C60(v16, v21, v24);
    *(_QWORD *)(v16 + 72) = sub_A7B660((__int64 *)(v16 + 72), v25, v30, 1, v23);
  }
  if ( a7 )
    sub_B99FD0(v16, 1, a7);
  if ( a8 )
    sub_B99FD0(v16, 7, a8);
  if ( a9 )
    sub_B99FD0(v16, 8, a9);
  return v16;
}
