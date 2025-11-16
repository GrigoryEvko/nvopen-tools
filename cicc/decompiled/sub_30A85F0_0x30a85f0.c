// Function: sub_30A85F0
// Address: 0x30a85f0
//
_QWORD *__fastcall sub_30A85F0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // rax
  bool v8; // cc
  _QWORD *v9; // rax
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  int v13; // [rsp+Ch] [rbp-64h] BYREF
  _QWORD v14[12]; // [rsp+10h] [rbp-60h] BYREF

  result = (_QWORD *)sub_30A7D00(a1);
  if ( result )
  {
    v5 = (__int64)result;
    v6 = sub_AA4B30(a1[5]);
    v7 = sub_B59BC0(v5);
    v8 = *(_DWORD *)(v7 + 32) <= 0x40u;
    v9 = *(_QWORD **)(v7 + 24);
    if ( !v8 )
      v9 = (_QWORD *)*v9;
    v13 = (int)v9;
    v10 = sub_B491C0((__int64)a1);
    v14[1] = a2;
    v14[2] = v6;
    v14[0] = &v13;
    v14[3] = a3;
    v14[4] = a1;
    return sub_30A7F20(a2, (__int64 (__fastcall *)(__int64, __int64))sub_30AABF0, (__int64)v14, v10, v11, v12);
  }
  return result;
}
