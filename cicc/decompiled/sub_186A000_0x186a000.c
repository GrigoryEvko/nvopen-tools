// Function: sub_186A000
// Address: 0x186a000
//
__int64 __fastcall sub_186A000(__int64 a1, int a2, char a3)
{
  __int64 v3; // rbp
  __int64 result; // rax
  unsigned __int8 (__fastcall **v5)(_QWORD, __int64); // [rsp-18h] [rbp-18h] BYREF
  int v6; // [rsp-10h] [rbp-10h]
  char v7; // [rsp-Ch] [rbp-Ch]
  __int64 v8; // [rsp-8h] [rbp-8h]

  result = *(unsigned int *)(a1 + 160);
  if ( a3 || *(_BYTE *)(a1 + 180) && *(_DWORD *)(a1 + 176) != (_DWORD)result )
  {
    v8 = v3;
    v5 = (unsigned __int8 (__fastcall **)(_QWORD, __int64))&off_49F1758;
    v7 = 1;
    v6 = result;
    return sub_16B38E0(
             (unsigned __int8 (__fastcall ***)(_QWORD, _QWORD))(a1 + 184),
             a1,
             &v5,
             (unsigned __int8 (__fastcall ***)(_QWORD, const char *))(a1 + 168),
             a2);
  }
  return result;
}
