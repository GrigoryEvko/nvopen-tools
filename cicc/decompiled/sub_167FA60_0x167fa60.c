// Function: sub_167FA60
// Address: 0x167fa60
//
__int64 __fastcall sub_167FA60(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 40);
  if ( (_DWORD)result == 2 )
    goto LABEL_8;
  if ( (unsigned int)result <= 2 )
  {
    if ( (_DWORD)result )
    {
      *(_QWORD *)(a1 + 32) = 4;
      return result;
    }
LABEL_8:
    *(_QWORD *)(a1 + 32) = 1;
    return result;
  }
  result = (unsigned int)(result - 3);
  if ( (unsigned int)result <= 1 )
    *(_QWORD *)(a1 + 32) = 0;
  return result;
}
