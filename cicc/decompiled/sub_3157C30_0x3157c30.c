// Function: sub_3157C30
// Address: 0x3157c30
//
__int64 __fastcall sub_3157C30(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  char v7; // al
  char v8; // al
  __int64 v9[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( (*(_BYTE *)(a2 + 35) & 4) != 0 )
    return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 128LL))(a1, a2, a3, a4);
  if ( *(_BYTE *)a2 == 3 )
  {
    v9[0] = *(_QWORD *)(a2 + 72);
    if ( (unsigned __int8)sub_A73380(v9, "bss-section", 0xBu) )
    {
      if ( (unsigned __int8)(a3 - 15) <= 2u )
        return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 128LL))(a1, a2, a3, a4);
    }
    v7 = sub_A73380(v9, "data-section", 0xCu);
    if ( (_BYTE)a3 == 19 )
    {
      if ( v7 )
        return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 128LL))(a1, a2, a3, a4);
    }
    v8 = sub_A73380(v9, "relro-section", 0xDu);
    if ( (_BYTE)a3 == 20 )
    {
      if ( v8 )
        return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 128LL))(a1, a2, a3, a4);
    }
    if ( (unsigned __int8)sub_A73380(v9, "rodata-section", 0xEu) && (unsigned __int8)(a3 - 4) <= 7u )
      return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 128LL))(a1, a2, a3, a4);
  }
  return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 264LL))(a1, a2, a3, a4);
}
