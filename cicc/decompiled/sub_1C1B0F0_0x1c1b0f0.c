// Function: sub_1C1B0F0
// Address: 0x1c1b0f0
//
__int64 __fastcall sub_1C1B0F0(__int64 a1, _DWORD *a2, __int64 a3, int a4)
{
  __int64 (__fastcall *v6)(__int64, __int64, _BOOL8); // r15
  char v7; // al
  _BOOL8 v8; // rdx
  __int64 result; // rax

  v6 = *(__int64 (__fastcall **)(__int64, __int64, _BOOL8))(*(_QWORD *)a1 + 168LL);
  v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v8 = 0;
  if ( v7 )
    v8 = *a2 == a4;
  result = v6(a1, a3, v8);
  if ( (_BYTE)result )
    *a2 = a4;
  return result;
}
