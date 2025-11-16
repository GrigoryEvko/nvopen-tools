// Function: sub_B8D270
// Address: 0xb8d270
//
__int64 __fastcall sub_B8D270(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int128 v16; // [rsp+20h] [rbp-50h] BYREF
  __int64 v17; // [rsp+30h] [rbp-40h]

  v7 = sub_BCB2E0(*a1);
  v16 = 0;
  v17 = 0;
  v8 = sub_ACD640(v7, a2, 0);
  *(_QWORD *)&v16 = sub_B8C140((__int64)a1, v8, v9, v10);
  v11 = sub_ACD640(v7, a3, 0);
  *((_QWORD *)&v16 + 1) = sub_B8C140((__int64)a1, v11, v12, v13);
  v17 = sub_B8C130(a1, a4, a5);
  return sub_B9C770(*a1, &v16, 3, 0, 1);
}
