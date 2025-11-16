// Function: sub_E5EB60
// Address: 0xe5eb60
//
bool __fastcall sub_E5EB60(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // rax
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  _BYTE v10[4]; // [rsp+4h] [rbp-2Ch] BYREF
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a1[1];
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 152LL);
  if ( v4 != sub_E5B850 && ((unsigned __int8 (__fastcall *)(__int64, __int64 *, __int64, _BYTE *))v4)(v3, a1, a2, v10) )
    return v10[0];
  v5 = *a1;
  v6 = *(_QWORD *)(a2 + 48);
  sub_E81940(*(_QWORD *)(a2 + 120), v11, a1);
  v7 = *(_QWORD *)(a2 + 112);
  v8 = v11[0];
  *(_QWORD *)(a2 + 48) = 0;
  *(_DWORD *)(a2 + 80) = 0;
  sub_E77860(
    v5,
    *((unsigned __int16 *)a1 + 36) | ((unsigned __int64)*((unsigned __int8 *)a1 + 74) << 16),
    v7,
    v8,
    a2 + 40);
  return *(_QWORD *)(a2 + 48) != v6;
}
