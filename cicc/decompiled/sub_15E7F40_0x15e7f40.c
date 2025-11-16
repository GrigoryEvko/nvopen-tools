// Function: sub_15E7F40
// Address: 0x15e7f40
//
_QWORD *__fastcall sub_15E7F40(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v9; // [rsp+8h] [rbp-48h] BYREF
  char v10; // [rsp+10h] [rbp-40h] BYREF
  __int16 v11; // [rsp+20h] [rbp-30h]

  v6 = a1[1];
  v9 = a2;
  v7 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(v6 + 56) + 40LL), 4, 0, 0);
  v11 = 257;
  return sub_15E6DE0(v7, (int)&v9, 1, a1, (int)&v10, 0, a3, a4);
}
