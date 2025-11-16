// Function: sub_38C6050
// Address: 0x38c6050
//
__int64 __fastcall sub_38C6050(__int64 a1, __int64 *a2, int a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // rax
  __int64 (__fastcall *v6)(__int64 *, __int64, _QWORD); // rbx
  __int64 v7; // rdx
  _BYTE v9[8]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE v10[64]; // [rsp+18h] [rbp-68h] BYREF
  char v11; // [rsp+58h] [rbp-28h]

  result = *(unsigned int *)(a1 + 128);
  if ( (_DWORD)result )
  {
    v5 = *a2;
    v11 = 0;
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(v5 + 160))(a2, a4, 0);
    v6 = *(__int64 (__fastcall **)(__int64 *, __int64, _QWORD))(*a2 + 176);
    sub_38C5CD0((__int64 *)a1, a2, (unsigned __int16)a3 | (BYTE2(a3) << 16), 0, 0, (__int64)v9);
    result = v6(a2, v7, 0);
    if ( v11 )
      return sub_167FA50((__int64)v10);
  }
  return result;
}
