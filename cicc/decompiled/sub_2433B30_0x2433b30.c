// Function: sub_2433B30
// Address: 0x2433b30
//
_QWORD *__fastcall sub_2433B30(__int64 *a1)
{
  _QWORD *v1; // r12
  __int64 v2; // rax
  __int64 v4; // [rsp+8h] [rbp-58h]
  _QWORD v5[4]; // [rsp+10h] [rbp-50h] BYREF
  char v6; // [rsp+30h] [rbp-30h]
  char v7; // [rsp+31h] [rbp-2Fh]

  v5[0] = "__hwasan_tls";
  v7 = 1;
  v6 = 3;
  BYTE4(v4) = 0;
  v1 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v1 )
    sub_B30000((__int64)v1, *(_QWORD *)(*a1 + 8), *(_QWORD **)(*a1 + 120), 0, 0, 0, (__int64)v5, 0, 3, v4, 0);
  v2 = *a1;
  v5[0] = v1;
  sub_2A41DC0(*(_QWORD *)(v2 + 8), v5, 1);
  return v1;
}
