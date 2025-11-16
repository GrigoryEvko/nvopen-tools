// Function: sub_71C980
// Address: 0x71c980
//
__int64 __fastcall sub_71C980(_DWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // r12

  result = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 24LL);
  if ( result )
  {
    v2 = *(_QWORD *)(result + 88);
    result = *(unsigned __int8 *)(v2 + 193);
    if ( (result & 0x10) != 0 )
    {
      if ( (*(_BYTE *)(v2 + 192) & 2) == 0 )
        return result;
    }
    else if ( (*(_BYTE *)(v2 + 206) & 8) == 0 || (*(_BYTE *)(v2 + 192) & 2) == 0 )
    {
      return result;
    }
    if ( (result & 0x20) == 0 && !*(_DWORD *)(v2 + 160) && !*(_QWORD *)(v2 + 344) )
    {
      if ( (a1[44] & 0x11000) == 0x1000 )
        return sub_71BE30(v2);
      result = sub_735B60(a1, 0);
      if ( !result || (*(_BYTE *)(result + 193) & 0x20) != 0 || *(_DWORD *)(result + 160) || *(_QWORD *)(result + 344) )
        return sub_71BE30(v2);
    }
  }
  return result;
}
