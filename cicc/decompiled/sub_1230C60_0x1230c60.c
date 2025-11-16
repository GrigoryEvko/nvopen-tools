// Function: sub_1230C60
// Address: 0x1230c60
//
__int64 __fastcall sub_1230C60(__int64 **a1, _QWORD *a2, __int64 *a3)
{
  unsigned int v3; // r12d
  _QWORD *v5; // rax
  _QWORD *v6; // r13
  __int64 v7; // [rsp+8h] [rbp-58h] BYREF
  _BYTE v8[32]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v9; // [rsp+30h] [rbp-30h]

  v3 = sub_122FE20(a1, &v7, a3);
  if ( (_BYTE)v3 )
    return v3;
  v9 = 257;
  v5 = sub_BD2C40(72, unk_3F10A14);
  v6 = v5;
  if ( v5 )
    sub_B549F0((__int64)v5, v7, (__int64)v8, 0, 0);
  *a2 = v6;
  return v3;
}
