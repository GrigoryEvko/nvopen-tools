// Function: sub_39F7ED0
// Address: 0x39f7ed0
//
__int64 __fastcall sub_39F7ED0(__int64 a1, int a2)
{
  __int64 result; // rax

  if ( a2 > 17 )
    goto LABEL_7;
  result = *(_QWORD *)(a1 + 8LL * a2);
  if ( (*(_BYTE *)(a1 + 199) & 0x40) == 0 || !*(_BYTE *)(a1 + a2 + 216) )
  {
    if ( byte_5057700[a2] == 8 )
      return *(_QWORD *)result;
LABEL_7:
    abort();
  }
  return result;
}
