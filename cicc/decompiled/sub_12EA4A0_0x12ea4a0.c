// Function: sub_12EA4A0
// Address: 0x12ea4a0
//
__int64 __fastcall sub_12EA4A0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  int v4; // ecx
  int v5; // edx
  int v6; // esi
  __int64 (__fastcall *v7)(__int64 *, __int64, _QWORD); // rbx
  __int64 v8; // r13
  _BYTE v10[16]; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v11)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-30h]

  v3 = *a2;
  v4 = *(unsigned __int8 *)(a1 + 22);
  v5 = *(unsigned __int8 *)(a1 + 21);
  v6 = *(unsigned __int8 *)(a1 + 20);
  v7 = *(__int64 (__fastcall **)(__int64 *, __int64, _QWORD))(v3 + 16);
  LODWORD(v3) = *(unsigned __int8 *)(a1 + 25);
  v11 = 0;
  v8 = sub_1A62BF0(
         *(_DWORD *)(a1 + 16),
         v6,
         v5,
         v4,
         *(unsigned __int8 *)(a1 + 23),
         *(unsigned __int8 *)(a1 + 24),
         v3,
         (__int64)v10);
  if ( v11 )
    v11(v10, v10, 3);
  return v7(a2, v8, 0);
}
