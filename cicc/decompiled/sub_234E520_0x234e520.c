// Function: sub_234E520
// Address: 0x234e520
//
_QWORD *__fastcall sub_234E520(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v5; // r8
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  _QWORD *v15; // rbx
  _BYTE v17[688]; // [rsp+0h] [rbp-590h] BYREF
  _QWORD v18[92]; // [rsp+2B0h] [rbp-2E0h] BYREF

  v18[0] = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, (__int64)a3) + 8);
  sub_D2ED00((__int64)v17, a3, (__int64 (__fastcall *)(__int64, __int64))sub_22FF990, (__int64)v18, v5);
  sub_D241E0((__int64)v18, (__int64)v17, v6, v7, v8, v9);
  v10 = (_QWORD *)sub_22077B0(0x2B8u);
  v15 = v10;
  if ( v10 )
  {
    *v10 = &unk_4A0B358;
    sub_D241E0((__int64)(v10 + 1), (__int64)v18, v11, v12, v13, v14);
  }
  sub_234E270((__int64)v18);
  *a1 = v15;
  sub_234E270((__int64)v17);
  return a1;
}
