// Function: sub_68A7B0
// Address: 0x68a7b0
//
__int64 __fastcall sub_68A7B0(__int64 a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  __int64 v3; // r13

  v1 = *(_QWORD *)(a1 + 8);
  if ( v1 && (*(_BYTE *)(a1 + 41) & 2) == 0 )
  {
    result = sub_6E2920(*(_QWORD *)(a1 + 8));
    if ( !*(_QWORD *)(v1 + 16) && (*(_BYTE *)(a1 + 40) & 8) == 0 )
    {
      result = sub_730250(v1);
      v3 = result;
      if ( result )
      {
        sub_6E2AC0(result);
        result = *(unsigned __int8 *)(v1 + 50);
        if ( (result & 0x10) != 0 )
        {
          *(_BYTE *)(v3 + 168) |= 0x28u;
          result = *(unsigned __int8 *)(v1 + 50);
        }
        if ( (result & 0x80u) != 0LL || (*(_BYTE *)(v3 + 170) & 0x40) != 0 )
          *(_BYTE *)(a1 + 41) |= 0x10u;
        *(_QWORD *)a1 = v3;
        *(_QWORD *)(a1 + 8) = 0;
      }
    }
  }
  else
  {
    *(_BYTE *)(a1 + 41) |= 2u;
    return sub_6E2A90();
  }
  return result;
}
