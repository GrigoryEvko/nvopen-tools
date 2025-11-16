// Function: sub_AA5FF0
// Address: 0xaa5ff0
//
__int64 __fastcall sub_AA5FF0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi

  for ( result = a1; ; result = *(_QWORD *)(result + 8) )
  {
    if ( !result )
      BUG();
    if ( *(_BYTE *)(result - 24) != 85 )
      break;
    v2 = *(_QWORD *)(result - 56);
    if ( !v2
      || *(_BYTE *)v2
      || *(_QWORD *)(v2 + 24) != *(_QWORD *)(result + 56)
      || (*(_BYTE *)(v2 + 33) & 0x20) == 0
      || (unsigned int)(*(_DWORD *)(v2 + 36) - 68) > 3 )
    {
      break;
    }
  }
  return result;
}
