// Function: sub_1601A30
// Address: 0x1601a30
//
__int64 __fastcall sub_1601A30(__int64 a1, char a2)
{
  __int64 result; // rax
  unsigned __int8 *v3; // rcx

  result = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( result )
  {
    v3 = *(unsigned __int8 **)(result + 24);
    result = 0;
    if ( (unsigned int)*v3 - 1 <= 1 )
      return *((_QWORD *)v3 + 17);
  }
  else if ( !a2 )
  {
    BUG();
  }
  return result;
}
