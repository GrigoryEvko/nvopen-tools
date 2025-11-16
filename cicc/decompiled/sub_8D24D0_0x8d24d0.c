// Function: sub_8D24D0
// Address: 0x8d24d0
//
__int64 __fastcall sub_8D24D0(__int64 a1)
{
  char v1; // cl
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 + 140);
  if ( v1 == 11 || (result = 0, (*(_BYTE *)(*(_QWORD *)(a1 + 168) + 110LL) & 4) != 0) )
  {
    result = **(_QWORD **)(*(_QWORD *)a1 + 96LL);
    if ( result )
    {
      while ( *(_BYTE *)(result + 80) != 8 || v1 != 11 && (*(_BYTE *)(*(_QWORD *)(result + 104) + 28LL) & 2) == 0 )
      {
        result = *(_QWORD *)(result + 16);
        if ( !result )
          return result;
      }
      return 1;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
