// Function: sub_E8DC70
// Address: 0xe8dc70
//
__int64 *__fastcall sub_E8DC70(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9

  sub_E98820(a1, a2, a3);
  sub_E5CB20(a1[37], a2, v3, v4, v5, v6);
  v7 = sub_E8BB10(a1, 0);
  *(_QWORD *)a2 = v7;
  *(_QWORD *)(a2 + 24) = *(_QWORD *)(v7 + 48);
  *(_BYTE *)(a2 + 9) = *(_BYTE *)(a2 + 9) & 0x8F | 0x10;
  return sub_E8DAF0((__int64)a1, a2, v8, v9, v10, v11);
}
