// Function: sub_6F6EB0
// Address: 0x6f6eb0
//
__int64 __fastcall sub_6F6EB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  __int64 result; // rax

  v6 = *(_BYTE *)(a1 + 16);
  if ( v6 == 2 )
    return sub_6ED0D0(a1, a2, a3, a4, a5, a6);
  if ( v6 > 2u )
  {
    if ( v6 != 5 )
      sub_721090(a1);
    return sub_6F6DD0(*(_QWORD *)(a1 + 144));
  }
  else if ( v6 )
  {
    if ( (*(_BYTE *)(a1 + 20) & 0x10) != 0 && !(_DWORD)a2 )
      sub_6F9060(a1);
    return *(_QWORD *)(a1 + 144);
  }
  else
  {
    result = sub_7305B0(a1, a2);
    *(_QWORD *)(result + 28) = *(_QWORD *)(a1 + 68);
  }
  return result;
}
