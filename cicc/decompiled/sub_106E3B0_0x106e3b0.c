// Function: sub_106E3B0
// Address: 0x106e3b0
//
__int64 __fastcall sub_106E3B0(__int64 a1, int a2, char a3)
{
  __int64 v3; // rbp
  __int64 result; // rax
  unsigned __int8 (__fastcall **v5)(_QWORD, __int64); // [rsp-18h] [rbp-18h] BYREF
  int v6; // [rsp-10h] [rbp-10h]
  char v7; // [rsp-Ch] [rbp-Ch]
  __int64 v8; // [rsp-8h] [rbp-8h]

  result = *(unsigned int *)(a1 + 136);
  if ( a3 || !*(_BYTE *)(a1 + 156) || *(_DWORD *)(a1 + 152) != (_DWORD)result )
  {
    v8 = v3;
    v5 = (unsigned __int8 (__fastcall **)(_QWORD, __int64))&off_49E5F18;
    v7 = 1;
    v6 = result;
    return sub_C55D10(
             (unsigned __int8 (__fastcall ***)(_QWORD, _QWORD))(a1 + 160),
             a1,
             &v5,
             (unsigned __int8 (__fastcall ***)(_QWORD, __int64))(a1 + 144),
             a2);
  }
  return result;
}
