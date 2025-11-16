// Function: sub_394B210
// Address: 0x394b210
//
__int64 __fastcall sub_394B210(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  char v6; // al
  bool v7; // al
  __int64 v9[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( (*(_BYTE *)(a2 + 34) & 0x20) == 0 )
  {
    v6 = *(_BYTE *)(a2 + 16);
    if ( v6 == 3 )
    {
      v9[0] = *(_QWORD *)(a2 + 72);
      if ( sub_155F280(v9, "bss-section", 0xBu) && (unsigned __int8)(a3 - 13) <= 2u )
        return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 72LL))(a1, a2, a3, a4);
      v7 = sub_155F280(v9, "data-section", 0xCu);
      if ( (_BYTE)a3 == 17 && v7 )
        return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 72LL))(a1, a2, a3, a4);
      if ( sub_155F280(v9, "rodata-section", 0xEu) && (unsigned __int8)(a3 - 3) <= 7u )
        return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 72LL))(a1, a2, a3, a4);
      v6 = *(_BYTE *)(a2 + 16);
    }
    if ( v6 || !sub_15602E0((_QWORD *)(a2 + 112), "implicit-section-name", 0x15u) )
      return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 152LL))(a1, a2, a3, a4);
  }
  return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 72LL))(a1, a2, a3, a4);
}
