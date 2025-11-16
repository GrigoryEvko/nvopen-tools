// Function: sub_2A00670
// Address: 0x2a00670
//
__int64 __fastcall sub_2A00670(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  bool v5; // zf
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // r15
  unsigned __int16 v12; // r14
  _QWORD *v13; // rdi
  __int64 v15; // [rsp+0h] [rbp-70h]
  __int64 v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+8h] [rbp-68h]
  _BYTE *v18; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int16 v19; // [rsp+18h] [rbp-58h]
  __int16 v20; // [rsp+30h] [rbp-40h]

  v5 = *a4 == 0;
  v6 = *(_QWORD *)(a2 + 8);
  v20 = 257;
  v7 = *a1;
  if ( !v5 )
  {
    v18 = a4;
    LOBYTE(v20) = 3;
  }
  v8 = a1[1];
  v15 = v7;
  v16 = v6;
  v9 = sub_22077B0(0x50u);
  v10 = v9;
  if ( v9 )
    sub_AA4D50(v9, v8, (__int64)&v18, v15, v16);
  sub_B43C20((__int64)&v18, v10);
  v11 = *(_QWORD *)(a2 + 8);
  v12 = v19;
  v17 = (__int64)v18;
  v13 = sub_BD2C40(72, 1u);
  if ( v13 )
    sub_B4C8F0((__int64)v13, v11, 1u, v17, v12);
  sub_AA5D60(*(_QWORD *)(a2 + 8), a3, v10);
  return v10;
}
