// Function: sub_AA5030
// Address: 0xaa5030
//
__int64 __fastcall sub_AA5030(__int64 a1, char a2)
{
  __int64 result; // rax
  __int64 v3; // rdi
  char v4; // dl
  __int64 v5; // rcx
  __int64 v6; // r8

  result = *(_QWORD *)(a1 + 56);
  v3 = a1 + 48;
  if ( result == v3 )
    return v3;
  while ( 1 )
  {
    if ( !result )
      BUG();
    v4 = *(_BYTE *)(result - 24);
    if ( v4 != 84 )
    {
      if ( v4 != 85 )
        break;
      v5 = *(_QWORD *)(result - 56);
      if ( !v5 || *(_BYTE *)v5 )
        break;
      v6 = *(_QWORD *)(v5 + 24);
      if ( (v6 != *(_QWORD *)(result + 56)
         || (*(_BYTE *)(v5 + 33) & 0x20) == 0
         || (unsigned int)(*(_DWORD *)(v5 + 36) - 68) > 3)
        && (!a2 || v6 != *(_QWORD *)(result + 56) || (*(_BYTE *)(v5 + 33) & 0x20) == 0 || *(_DWORD *)(v5 + 36) != 291) )
      {
        break;
      }
    }
    result = *(_QWORD *)(result + 8);
    if ( v3 == result )
      return v3;
  }
  return result;
}
