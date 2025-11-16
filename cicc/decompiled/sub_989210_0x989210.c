// Function: sub_989210
// Address: 0x989210
//
__int64 __fastcall sub_989210(_DWORD *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  char v7; // dl

  result = 0;
  v5 = a3;
  if ( (*a1 & 0x40) == 0 )
  {
    result = 1;
    if ( (*a1 & 0x90) != 0 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
        v5 = **(_QWORD **)(a3 + 16);
      v6 = sub_BCAC60(v5);
      v7 = (unsigned __int16)sub_B2DB90(a2, v6) >> 8;
      result = 1;
      if ( v7 )
      {
        result = 0;
        if ( v7 == 1 )
          return ((unsigned __int8)(*a1 >> 7) ^ 1) & 1;
      }
    }
  }
  return result;
}
