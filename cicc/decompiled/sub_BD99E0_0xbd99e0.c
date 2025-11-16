// Function: sub_BD99E0
// Address: 0xbd99e0
//
__int64 __fastcall sub_BD99E0(__int64 a1)
{
  __int64 v1; // r8
  __int64 result; // rax
  __int16 v3; // dx

  if ( *(_BYTE *)a1 == 34 )
  {
    v1 = *(_QWORD *)(a1 - 64);
  }
  else
  {
    v3 = *(_WORD *)(a1 + 2);
    v1 = 0;
    if ( *(_BYTE *)a1 == 39 )
    {
      if ( (v3 & 1) != 0 )
        v1 = *(_QWORD *)(*(_QWORD *)(a1 - 8) + 32LL);
    }
    else if ( (v3 & 1) != 0 )
    {
      v1 = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
    }
  }
  result = sub_AA4FF0(v1);
  if ( result )
    result -= 24;
  return result;
}
