// Function: sub_1ACEAA0
// Address: 0x1aceaa0
//
__int64 __fastcall sub_1ACEAA0(__int64 a1, __int64 a2, char a3)
{
  __int64 result; // rax

  result = *(_BYTE *)(a2 + 32) & 0xF;
  if ( *(_BYTE *)(a1 + 24) )
  {
    if ( (unsigned int)(result - 7) <= 1 )
    {
      if ( a3 )
        return 0;
    }
  }
  else if ( *(_QWORD *)(a1 + 16) )
  {
    __asm { jmp     rcx }
  }
  return result;
}
