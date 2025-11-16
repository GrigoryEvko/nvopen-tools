// Function: sub_B35570
// Address: 0xb35570
//
__int64 __fastcall sub_B35570(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        unsigned __int16 a8)
{
  __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // r14d
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 *v17; // rax
  __int64 v22; // [rsp+20h] [rbp-60h] BYREF
  __int64 v23; // [rsp+28h] [rbp-58h]
  __int64 v24; // [rsp+30h] [rbp-50h] BYREF
  __int64 v25; // [rsp+38h] [rbp-48h]
  __int64 v26; // [rsp+40h] [rbp-40h]

  v10 = a7;
  v11 = (unsigned __int8)a8;
  if ( !HIBYTE(a8) )
    v11 = *(unsigned __int8 *)(a1 + 109);
  sub_E3F8A0(&v24, v11);
  v12 = sub_B9B140(*(_QWORD *)(a1 + 72), v24, v25);
  v13 = sub_B9F6F0(*(_QWORD *)(a1 + 72), v12);
  v14 = *(_DWORD *)(a1 + 104);
  if ( BYTE4(a5) )
    v14 = a5;
  BYTE4(v23) = 0;
  v26 = v13;
  v15 = *(_QWORD *)(a3 + 8);
  v25 = a4;
  v24 = a3;
  v22 = v15;
  v16 = sub_B33D10(a1, a2, (__int64)&v22, 1, (int)&v24, 3, v23, a6);
  v17 = (__int64 *)sub_BD5C60(v16, a2);
  *(_QWORD *)(v16 + 72) = sub_A7A090((__int64 *)(v16 + 72), v17, -1, 72);
  if ( a7 || (v10 = *(_QWORD *)(a1 + 96)) != 0 )
    sub_B99FD0(v16, 3, v10);
  sub_B45150(v16, v14);
  return v16;
}
