// Function: sub_25AC600
// Address: 0x25ac600
//
__int64 __fastcall sub_25AC600(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // rax
  unsigned __int8 *v3; // rcx
  __int64 result; // rax

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
    v2 = *(_QWORD *)(a1 - 32);
  else
    v2 = a1 - 8LL * ((v1 >> 2) & 0xF) - 16;
  v3 = *(unsigned __int8 **)(v2 + 8);
  result = 0;
  if ( (unsigned int)*v3 - 1 <= 1 )
  {
    result = *((_QWORD *)v3 + 17);
    if ( result )
    {
      if ( *(_BYTE *)result == 17 )
      {
        if ( *(_DWORD *)(result + 32) != 64 )
          return 0;
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
