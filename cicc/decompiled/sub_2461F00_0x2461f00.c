// Function: sub_2461F00
// Address: 0x2461f00
//
_QWORD *__fastcall sub_2461F00(__int64 *a1, char *a2, signed __int64 a3)
{
  __int64 v3; // rax
  _QWORD *v4; // r14
  __int64 v5; // rbx
  _QWORD *v6; // r12
  __int64 v8; // [rsp+8h] [rbp-58h]
  _BYTE v9[32]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v10; // [rsp+30h] [rbp-30h]

  v3 = sub_AC9B20(*a1, a2, a3, 1);
  BYTE4(v8) = 0;
  v4 = *(_QWORD **)(v3 + 8);
  v5 = v3;
  v10 = 257;
  v6 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v6 )
    sub_B30000((__int64)v6, (__int64)a1, v4, 1, 8, v5, (__int64)v9, 0, 0, v8, 0);
  return v6;
}
