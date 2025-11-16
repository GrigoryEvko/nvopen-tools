// Function: sub_D00280
// Address: 0xd00280
//
char __fastcall sub_D00280(__int64 a1, __int64 a2)
{
  char result; // al

  result = 0;
  if ( *(_QWORD *)(*(_QWORD *)a1 + 8LL) == *(_QWORD *)(*(_QWORD *)a2 + 8LL) )
  {
    if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)(a2 + 8) || (result = 1, *(_DWORD *)(a1 + 16) != *(_DWORD *)(a2 + 16)) )
    {
      if ( *(_BYTE *)(a1 + 20) || (result = *(_BYTE *)(a2 + 20)) != 0 )
      {
        result = 0;
        if ( *(_DWORD *)(a1 + 8) + *(_DWORD *)(a1 + 12) == *(_DWORD *)(a2 + 8) + *(_DWORD *)(a2 + 12) )
          return *(_DWORD *)(a1 + 16) == *(_DWORD *)(a2 + 16);
      }
    }
  }
  return result;
}
