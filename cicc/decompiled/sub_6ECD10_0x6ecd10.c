// Function: sub_6ECD10
// Address: 0x6ecd10
//
_BOOL8 __fastcall sub_6ECD10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BOOL8 result; // rax
  __int64 v7; // r12
  __int64 (__fastcall *v8)(_QWORD *, _DWORD *); // [rsp-E8h] [rbp-E8h] BYREF
  int v9; // [rsp-98h] [rbp-98h]
  int v10; // [rsp-88h] [rbp-88h]

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 1 )
  {
    v7 = *(_QWORD *)(a1 + 144);
    if ( (*(_BYTE *)(v7 + 25) & 3) != 0 )
    {
      sub_76C7C0(&v8, a2, a3, a4, a5, a6);
      v8 = sub_6DF820;
      v10 = 1;
      sub_76CDC0(v7);
      return v9 != 0;
    }
  }
  return result;
}
