// Function: sub_2C25A30
// Address: 0x2c25a30
//
__int64 __fastcall sub_2C25A30(__int64 a1)
{
  __int64 result; // rax
  __int64 *v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rsi

  result = 0;
  if ( *(_DWORD *)(a1 + 88) == 2 )
  {
    v2 = *(__int64 **)(a1 + 80);
    v3 = *v2;
    result = v2[1];
    if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 1 > 1 )
      return 0;
    if ( (unsigned __int8)(*(_BYTE *)(result + 8) - 1) > 1u )
      return 0;
    v4 = *(unsigned int *)(v3 + 88);
    if ( *(unsigned int *)(result + 88) + v4 != 1 )
      return 0;
    if ( (_DWORD)v4 == 1 && **(_QWORD **)(v3 + 80) == result )
      return v3;
    if ( *(_DWORD *)(result + 88) == 1 )
    {
      if ( v3 != **(_QWORD **)(result + 80) )
        return 0;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
