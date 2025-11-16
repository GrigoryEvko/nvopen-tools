// Function: sub_650FD0
// Address: 0x650fd0
//
__int64 __fastcall sub_650FD0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 200);
  if ( result && dword_4D043E0 && !(_DWORD)qword_4F077B4 )
  {
    while ( *(_BYTE *)(result + 9) != 2 && (*(_BYTE *)(result + 11) & 0x10) == 0 || *(_BYTE *)(result + 10) != 10 )
    {
      result = *(_QWORD *)result;
      if ( !result )
        return result;
    }
    return sub_6851C0(1113, *(_QWORD *)(result + 40));
  }
  return result;
}
