// Function: sub_23980A0
// Address: 0x23980a0
//
_QWORD *__fastcall sub_23980A0(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 a4)
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
  _BYTE v15[592]; // [rsp+0h] [rbp-4C0h] BYREF
  _BYTE v16[624]; // [rsp+250h] [rbp-270h] BYREF

  sub_D83BE0((__int64)v15, a2 + 8, a3, a4);
  sub_D77AB0((__int64)v16, (__int64)v15, v4, v5, v6, v7);
  v8 = (_QWORD *)sub_22077B0(0x250u);
  v13 = v8;
  if ( v8 )
  {
    *v8 = &unk_4A0B330;
    sub_D77AB0((__int64)(v8 + 1), (__int64)v16, v9, v10, v11, v12);
  }
  sub_9CD560((__int64)v16);
  *a1 = v13;
  sub_9CD560((__int64)v15);
  return a1;
}
