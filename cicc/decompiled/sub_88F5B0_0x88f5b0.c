// Function: sub_88F5B0
// Address: 0x88f5b0
//
__int64 __fastcall sub_88F5B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r9
  int v4; // r13d

  result = *(_QWORD *)(a2 + 24);
  if ( result && (*(_BYTE *)(result + 82) & 4) != 0 )
  {
    result = sub_87DC80(a2, 0, 0, 1);
    if ( (*(_BYTE *)(a2 + 17) & 0x20) != 0 )
      return result;
  }
  else if ( (*(_BYTE *)(a2 + 17) & 0x20) != 0 )
  {
    return result;
  }
  result = sub_8D2310(*(_QWORD *)(a1 + 288));
  v3 = *(_QWORD *)(a1 + 8);
  v4 = result;
  if ( (v3 & 1) == 0 )
  {
    result = sub_8D2310(*(_QWORD *)(a1 + 288));
    v3 = *(_QWORD *)(a1 + 8);
    if ( !(_DWORD)result
      || (v3 & 0xC00) == 0
      && (*(_BYTE *)(a1 + 16) & 0x30) == 0
      && (*(_BYTE *)(a2 + 16) & 0x10) == 0
      && ((v3 & 8) == 0
       || (*(_BYTE *)(a2 + 18) & 2) == 0
       || (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0
       || (result = *(_QWORD *)(a2 + 32), (*(_BYTE *)(result + 177) & 0x20) == 0)) )
    {
      result = sub_64E990(a1 + 32, *(_QWORD *)(a1 + 288), v4, 0, 0, (BYTE1(v3) ^ 1) & 1);
      v3 = *(_QWORD *)(a1 + 8);
    }
  }
  if ( (v3 & 0x20) != 0 )
    return sub_6851C0(0xFFu, (_DWORD *)(a1 + 32));
  return result;
}
