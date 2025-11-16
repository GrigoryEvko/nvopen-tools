// Function: sub_1479810
// Address: 0x1479810
//
__int64 __fastcall sub_1479810(__int64 a1, __int64 a2, unsigned int a3, bool *a4)
{
  unsigned int v6; // r13d
  __int64 result; // rax
  __int64 v8; // r15

  if ( a3 <= 0x25 )
  {
    if ( a3 > 0x21 && (*(_BYTE *)(a2 + 26) & 2) != 0 )
    {
      *a4 = a3 - 34 <= 1;
      return 1;
    }
  }
  else
  {
    v6 = a3 - 38;
    result = 0;
    if ( a3 - 38 > 3 )
      return result;
    if ( (*(_BYTE *)(a2 + 26) & 4) != 0 )
    {
      v8 = sub_13A5BC0((_QWORD *)a2, a1);
      result = sub_1477BC0(a1, v8);
      if ( (_BYTE)result )
      {
        *a4 = v6 <= 1;
        return result;
      }
      result = sub_1477A90(a1, v8);
      if ( (_BYTE)result )
      {
        *a4 = a3 - 40 <= 1;
        return result;
      }
    }
  }
  return 0;
}
