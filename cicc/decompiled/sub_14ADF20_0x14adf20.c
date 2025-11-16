// Function: sub_14ADF20
// Address: 0x14adf20
//
__int64 __fastcall sub_14ADF20(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdi
  unsigned __int8 v3; // al
  __int64 result; // rax
  __int64 v5; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( !v1 )
    return 1;
  while ( 1 )
  {
    v2 = sub_1648700(v1);
    v3 = *(_BYTE *)(v2 + 16);
    if ( v3 <= 0x17u )
      break;
    if ( v3 == 72 )
    {
      result = sub_14ADF20(v2);
      if ( !(_BYTE)result )
        return result;
    }
    else
    {
      if ( v3 != 78 )
        return 0;
      v5 = *(_QWORD *)(v2 - 24);
      if ( *(_BYTE *)(v5 + 16) || (*(_BYTE *)(v5 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v5 + 36) - 116) > 1 )
        return 0;
    }
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return 1;
  }
  return 0;
}
