// Function: sub_2399EB0
// Address: 0x2399eb0
//
_QWORD *__fastcall sub_2399EB0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // rbx
  _BYTE v15[1024]; // [rsp+0h] [rbp-820h] BYREF
  _BYTE v16[1056]; // [rsp+400h] [rbp-420h] BYREF

  sub_102CBB0((__int64)v15, (int *)(a2 + 8), a3, a4);
  sub_2399840((__int64)v16, (__int64)v15, v4, v5, v6, v7);
  v8 = (_QWORD *)sub_22077B0(0x400u);
  v13 = v8;
  if ( v8 )
  {
    *v8 = &unk_4A0AF48;
    sub_2399840((__int64)(v8 + 1), (__int64)v16, v9, v10, v11, v12);
  }
  sub_102BD40((__int64)v16);
  *a1 = v13;
  sub_102BD40((__int64)v15);
  return a1;
}
