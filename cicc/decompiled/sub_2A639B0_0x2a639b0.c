// Function: sub_2A639B0
// Address: 0x2a639b0
//
bool __fastcall sub_2A639B0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool result; // al
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  bool v10; // [rsp+Fh] [rbp-21h]

  result = sub_2A625F0((__int64)a2, a4, a5, SBYTE1(a5), HIDWORD(a5));
  if ( result )
  {
    v10 = result;
    sub_2A62F90(a1, a2, a3, v7, v8, v9);
    return v10;
  }
  return result;
}
