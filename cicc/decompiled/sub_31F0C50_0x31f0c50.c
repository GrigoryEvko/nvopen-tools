// Function: sub_31F0C50
// Address: 0x31f0c50
//
__int64 __fastcall sub_31F0C50(__int64 a1, int a2)
{
  __int64 result; // rax

  if ( a2 == 255 )
    return 0;
  result = a2 & 7;
  if ( (_DWORD)result == 3 )
    return 4;
  if ( (a2 & 4) != 0 )
  {
    if ( (_DWORD)result == 4 )
      return 8;
    goto LABEL_12;
  }
  if ( (a2 & 7) == 0 )
    return *(unsigned int *)(*(_QWORD *)(a1 + 208) + 8LL);
  if ( (_DWORD)result != 2 )
LABEL_12:
    BUG();
  return result;
}
