// Function: sub_7EBC30
// Address: 0x7ebc30
//
void *__fastcall sub_7EBC30(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r12
  __int64 *v3; // rax
  _BYTE *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // rax
  unsigned __int16 v10; // r13
  __int64 *v11; // rbx
  _BYTE *v12; // r12
  _BYTE *v13; // r12
  void *result; // rax

  v2 = sub_72B0F0(a1, 0);
  v3 = (__int64 *)sub_73DCD0(a2);
  v4 = sub_7E45A0(v3);
  v9 = sub_731370((__int64)v4, 0, v5, v6, v7, v8);
  v10 = *(_WORD *)(v2 + 224);
  v11 = v9;
  v12 = sub_724D50(1);
  sub_72BAF0((__int64)v12, v10, 5u);
  v13 = sub_7EBB70((__int64)v12);
  result = sub_73DBF0(0x32u, *v11, (__int64)v11);
  v11[2] = (__int64)v13;
  return result;
}
