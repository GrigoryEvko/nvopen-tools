// Function: sub_226AC00
// Address: 0x226ac00
//
__int64 __fastcall sub_226AC00(__int64 a1, int a2)
{
  unsigned __int8 (__fastcall **v3)(_QWORD, __int64); // [rsp+8h] [rbp-8h] BYREF

  v3 = (unsigned __int8 (__fastcall **)(_QWORD, __int64))&unk_4A083F8;
  return sub_C55D10(
           (unsigned __int8 (__fastcall ***)(_QWORD, _QWORD))(a1 + 160),
           a1,
           &v3,
           (unsigned __int8 (__fastcall ***)(_QWORD, __int64))(a1 + 152),
           a2);
}
