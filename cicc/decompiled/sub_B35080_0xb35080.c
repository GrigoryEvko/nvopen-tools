// Function: sub_B35080
// Address: 0xb35080
//
__int64 __fastcall sub_B35080(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 *v14; // rax
  __int64 v15; // [rsp+10h] [rbp-78h] BYREF
  _QWORD v16[4]; // [rsp+18h] [rbp-70h] BYREF
  _DWORD v17[8]; // [rsp+38h] [rbp-50h] BYREF
  __int16 v18; // [rsp+58h] [rbp-30h]

  v6 = *(_QWORD *)(a2 + 8);
  v16[0] = a2;
  v15 = v6;
  v16[1] = a3;
  v16[2] = a5;
  v18 = 257;
  v7 = sub_B34BE0(a1, 0xE1u, (int)v16, 3, (__int64)&v15, 1, (__int64)v17);
  v8 = v7;
  if ( BYTE1(a4) )
  {
    v10 = (__int64 *)sub_BD5C60(v7, 225);
    v11 = a4;
    v12 = sub_A77A40(v10, a4);
    v17[0] = 1;
    v13 = v12;
    v14 = (__int64 *)sub_BD5C60(v8, v11);
    *(_QWORD *)(v8 + 72) = sub_A7B660((__int64 *)(v8 + 72), v14, v17, 1, v13);
  }
  return v8;
}
