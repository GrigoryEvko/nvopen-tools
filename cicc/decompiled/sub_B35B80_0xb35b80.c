// Function: sub_B35B80
// Address: 0xb35b80
//
__int64 __fastcall sub_B35B80(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int16 a7)
{
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 *v19; // rax
  __int64 v21; // [rsp+0h] [rbp-70h]
  __int64 v23; // [rsp+10h] [rbp-60h] BYREF
  __int64 v24; // [rsp+18h] [rbp-58h]
  __int64 v25; // [rsp+20h] [rbp-50h] BYREF
  __int64 v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+30h] [rbp-40h]
  __int64 v28; // [rsp+38h] [rbp-38h]

  v10 = sub_B52C80(a3);
  v12 = sub_B9B140(*(_QWORD *)(a1 + 72), v10, v11);
  v13 = sub_B9F6F0(*(_QWORD *)(a1 + 72), v12);
  v14 = (unsigned __int8)a7;
  if ( !HIBYTE(a7) )
    v14 = *(unsigned __int8 *)(a1 + 109);
  v21 = v13;
  sub_E3F8A0(&v25, v14);
  v15 = sub_B9B140(*(_QWORD *)(a1 + 72), v25, v26);
  v16 = sub_B9F6F0(*(_QWORD *)(a1 + 72), v15);
  BYTE4(v24) = 0;
  v28 = v16;
  v17 = *(_QWORD *)(a4 + 8);
  v27 = v21;
  v25 = a4;
  v26 = a5;
  v23 = v17;
  v18 = sub_B33D10(a1, a2, (__int64)&v23, 1, (int)&v25, 4, v24, a6);
  v19 = (__int64 *)sub_BD5C60(v18, a2);
  *(_QWORD *)(v18 + 72) = sub_A7A090((__int64 *)(v18 + 72), v19, -1, 72);
  return v18;
}
