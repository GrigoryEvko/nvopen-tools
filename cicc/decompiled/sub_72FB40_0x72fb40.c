// Function: sub_72FB40
// Address: 0x72fb40
//
__int64 __fastcall sub_72FB40(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rdi

  result = *(unsigned __int8 *)(a1 + 173);
  if ( (_BYTE)result == 12 || (*(_BYTE *)(a1 + 171) & 4) != 0 )
  {
    a2[20] = 1;
    a2[18] = 1;
    return result;
  }
  if ( (_BYTE)result == 6 )
  {
    if ( *(_BYTE *)(a1 + 176) == 1 )
    {
      v3 = *(_QWORD *)(a1 + 184);
      if ( (*(_BYTE *)(v3 + 89) & 4) == 0
        || (result = *(_QWORD *)(*(_QWORD *)(v3 + 40) + 32LL), (*(_BYTE *)(result + 177) & 0x20) == 0) )
      {
        result = sub_72FA80(v3, (__int64)a2);
        a2[19] = 1;
        return result;
      }
      a2[20] = 1;
      a2[18] = 1;
    }
LABEL_8:
    a2[19] = 1;
    return result;
  }
  result = (unsigned int)(result - 9);
  if ( (unsigned __int8)result > 2u )
    goto LABEL_8;
  return result;
}
