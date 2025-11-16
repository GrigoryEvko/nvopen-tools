// Function: sub_6EB5C0
// Address: 0x6eb5c0
//
__int64 __fastcall sub_6EB5C0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // [rsp+8h] [rbp-18h]

  if ( *(_BYTE *)(a1 + 16) == 1 )
  {
    result = sub_72B0F0(*(_QWORD *)(a1 + 144), 0);
  }
  else
  {
    if ( *(_WORD *)(a1 + 16) != 514
      || *(_BYTE *)(a1 + 317) != 6
      || *(_BYTE *)(a1 + 320)
      || *(_QWORD *)(a1 + 336)
      || (*(_BYTE *)(a1 + 312) & 8) != 0 )
    {
      return 0;
    }
    result = *(_QWORD *)(a1 + 328);
  }
  if ( !result )
    return 0;
  if ( (*(_BYTE *)(result + 207) & 0x30) == 0x10 )
  {
    v2 = result;
    sub_8B1A30(result, a1 + 68);
    return v2;
  }
  return result;
}
