// Function: sub_5EB240
// Address: 0x5eb240
//
__int64 __fastcall sub_5EB240(__int64 a1)
{
  __int64 v1; // r8
  __int64 result; // rax

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 88LL);
  result = *(unsigned __int8 *)(v1 + 193);
  *(_QWORD *)(v1 + 344) = 0;
  if ( (result & 0x20) == 0 && !*(_DWORD *)(v1 + 160) )
  {
    if ( !dword_4CF8020 || (result & 2) != 0 || (*(_BYTE *)(v1 + 207) & 0x10) != 0 )
    {
      if ( (*(_BYTE *)(v1 + 206) & 8) != 0 )
        return sub_71D150(v1);
      else
        return sub_5E6120(a1);
    }
    else
    {
      if ( !qword_4CF8018 )
        qword_4CF8018 = a1;
      result = qword_4CF8010;
      if ( qword_4CF8010 )
        *(_QWORD *)qword_4CF8010 = a1;
      qword_4CF8010 = a1;
    }
  }
  return result;
}
