// Function: sub_3215620
// Address: 0x3215620
//
__int64 __fastcall sub_3215620(__int64 a1, __int64 a2, __int16 a3)
{
  char v3; // al
  __int64 result; // rax

  if ( a3 == 7 )
    return 8;
  if ( a3 == 23 )
  {
    v3 = *(_BYTE *)(a2 + 3);
    if ( !v3 )
      return 4;
    if ( v3 == 1 )
      return 8;
LABEL_10:
    BUG();
  }
  result = 4;
  if ( a3 != 6 )
    goto LABEL_10;
  return result;
}
