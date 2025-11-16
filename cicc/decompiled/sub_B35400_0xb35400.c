// Function: sub_B35400
// Address: 0xb35400
//
__int64 __fastcall sub_B35400(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int16 a8,
        unsigned __int16 a9)
{
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // r15d
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 *v22; // rax
  __int64 v24; // [rsp+0h] [rbp-80h]
  __int64 v28; // [rsp+20h] [rbp-60h] BYREF
  __int64 v29; // [rsp+28h] [rbp-58h]
  __int64 v30; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+38h] [rbp-48h]
  __int64 v32; // [rsp+40h] [rbp-40h]
  __int64 v33; // [rsp+48h] [rbp-38h]

  v11 = *(unsigned __int8 *)(a1 + 110);
  v12 = a8;
  if ( HIBYTE(a8) )
    v11 = a8;
  LOWORD(v12) = HIBYTE(a8);
  sub_E3F6F0(&v30, v11, a3, v12);
  v13 = sub_B9B140(*(_QWORD *)(a1 + 72), v30, v31);
  v14 = sub_B9F6F0(*(_QWORD *)(a1 + 72), v13);
  v15 = (unsigned __int8)a9;
  if ( !HIBYTE(a9) )
    v15 = *(unsigned __int8 *)(a1 + 109);
  v24 = v14;
  sub_E3F8A0(&v30, v15);
  v16 = sub_B9B140(*(_QWORD *)(a1 + 72), v30, v31);
  v17 = sub_B9F6F0(*(_QWORD *)(a1 + 72), v16);
  v18 = *(_DWORD *)(a1 + 104);
  if ( BYTE4(a5) )
    v18 = a5;
  v19 = a2;
  BYTE4(v29) = 0;
  v33 = v17;
  v20 = *(_QWORD *)(a3 + 8);
  v31 = a4;
  v32 = v24;
  v30 = a3;
  v28 = v20;
  v21 = sub_B33D10(a1, a2, (__int64)&v28, 1, (int)&v30, 4, v29, a6);
  v22 = (__int64 *)sub_BD5C60(v21, v19);
  *(_QWORD *)(v21 + 72) = sub_A7A090((__int64 *)(v21 + 72), v22, -1, 72);
  if ( a7 || (a7 = *(_QWORD *)(a1 + 96)) != 0 )
    sub_B99FD0(v21, 3, a7);
  sub_B45150(v21, v18);
  return v21;
}
