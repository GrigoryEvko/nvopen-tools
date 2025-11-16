// Function: sub_E5F570
// Address: 0xe5f570
//
__int64 __fastcall sub_E5F570(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  _DWORD v9[10]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *a1;
  v5 = sub_E808D0(a2, 0, *a1, 0);
  v6 = sub_E808D0(a3, 0, v4, 0);
  v7 = sub_E81A00(18, v6, v5, v4, 0);
  sub_E81940(v7, v9, a1);
  return v9[0];
}
