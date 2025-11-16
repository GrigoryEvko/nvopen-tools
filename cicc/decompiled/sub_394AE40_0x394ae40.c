// Function: sub_394AE40
// Address: 0x394ae40
//
__int64 __fastcall sub_394AE40(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 (__fastcall *v5)(__int64, __int64, char); // rbx
  __int64 v6; // rax
  __int64 result; // rax
  _DWORD v8[5]; // [rsp+Ch] [rbp-14h] BYREF

  v3 = *a1;
  v4 = *(_QWORD *)(a2 + 40);
  v8[0] = 0;
  v5 = *(__int64 (__fastcall **)(__int64, __int64, char))(v3 + 40);
  v6 = sub_1632FA0(v4);
  if ( v5 != sub_394ACD0 )
    return ((__int64 (__fastcall *)(__int64 *, __int64, __int64, _QWORD, _DWORD *))v5)(a1, v6, 3, 0, v8);
  result = a1[7];
  if ( !result )
    return a1[5];
  return result;
}
