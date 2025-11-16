// Function: sub_75C030
// Address: 0x75c030
//
unsigned __int64 __fastcall sub_75C030(__int64 a1)
{
  unsigned __int64 result; // rax
  __int64 v3; // rdx
  __int64 *v4; // rdx
  __int64 v5; // rdx

  result = (unsigned int)dword_4F08010;
  while ( 1 )
  {
    while ( !(_DWORD)result || (*(_BYTE *)(a1 - 8) & 2) != 0 )
    {
      result = *(unsigned __int8 *)(a1 + 178);
      if ( (result & 0x40) != 0 )
        return result;
      *(_BYTE *)(a1 + 178) = result | 0x40;
      sub_75BF90(a1);
      sub_7607C0(a1, 6);
      result = *(_QWORD *)(a1 + 32);
      if ( !result )
        return result;
      v3 = *(_QWORD *)result;
      if ( a1 == *(_QWORD *)result || (*(_BYTE *)(v3 - 8) & 2) == 0 )
        return result;
      result = (unsigned int)dword_4F08010;
      a1 = v3;
    }
    v4 = *(__int64 **)(a1 + 32);
    if ( !v4 )
      break;
    v5 = *v4;
    if ( a1 == v5 || (*(_BYTE *)(v5 - 8) & 2) == 0 )
      break;
    a1 = v5;
  }
  return result;
}
