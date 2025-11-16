// Function: sub_B343C0
// Address: 0xb343c0
//
__int64 __fastcall sub_B343C0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        unsigned int a6,
        __int64 a7,
        unsigned __int8 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r12
  __int64 *v21; // rsi
  __int64 v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 v27; // rdx
  __int64 *v28; // rsi
  __int64 v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rdx
  __int64 *v35; // rax
  __int64 v36; // [rsp-10h] [rbp-E0h]
  unsigned int v39; // [rsp+28h] [rbp-A8h]
  _QWORD v40[4]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v41[4]; // [rsp+50h] [rbp-80h] BYREF
  _DWORD v42[8]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v43; // [rsp+90h] [rbp-40h]

  v41[1] = a5;
  v15 = *(_QWORD *)(a1 + 72);
  v41[0] = a3;
  v41[2] = a7;
  v16 = sub_BCB2A0(v15);
  a2 = (unsigned int)a2;
  v41[3] = sub_ACD640(v16, a8, 0);
  v40[0] = *(_QWORD *)(a3 + 8);
  v40[1] = *(_QWORD *)(a5 + 8);
  v40[2] = *(_QWORD *)(a7 + 8);
  v43 = 257;
  v17 = sub_B33D10(a1, a2, (__int64)v40, 3, (int)v41, 4, v39, (__int64)v42);
  v18 = v36;
  v19 = v17;
  if ( BYTE1(a4) )
  {
    v21 = (__int64 *)sub_BD5C60(v17, (unsigned int)a2, v36);
    *(_QWORD *)(v19 + 72) = sub_A7B980((__int64 *)(v19 + 72), v21, 1, 86);
    v23 = (__int64 *)sub_BD5C60(v19, v21, v22);
    v24 = a4;
    v25 = sub_A77A40(v23, a4);
    v42[0] = 0;
    v26 = v25;
    a2 = sub_BD5C60(v19, v24, v27);
    *(_QWORD *)(v19 + 72) = sub_A7B660((__int64 *)(v19 + 72), (__int64 *)a2, v42, 1, v26);
    if ( !BYTE1(a6) )
      goto LABEL_3;
  }
  else if ( !BYTE1(a6) )
  {
    goto LABEL_3;
  }
  v28 = (__int64 *)sub_BD5C60(v19, a2, v18);
  *(_QWORD *)(v19 + 72) = sub_A7B980((__int64 *)(v19 + 72), v28, 2, 86);
  v30 = (__int64 *)sub_BD5C60(v19, v28, v29);
  v31 = a6;
  v32 = sub_A77A40(v30, a6);
  v42[0] = 1;
  v33 = v32;
  v35 = (__int64 *)sub_BD5C60(v19, v31, v34);
  *(_QWORD *)(v19 + 72) = sub_A7B660((__int64 *)(v19 + 72), v35, v42, 1, v33);
LABEL_3:
  if ( a9 )
    sub_B99FD0(v19, 1, a9);
  if ( a10 )
    sub_B99FD0(v19, 5, a10);
  if ( a11 )
    sub_B99FD0(v19, 7, a11);
  if ( a12 )
    sub_B99FD0(v19, 8, a12);
  return v19;
}
