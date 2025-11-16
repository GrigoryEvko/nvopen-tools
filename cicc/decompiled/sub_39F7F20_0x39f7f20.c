// Function: sub_39F7F20
// Address: 0x39f7f20
//
__int64 __fastcall sub_39F7F20(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax

  if ( a2 > 17 )
    goto LABEL_7;
  result = (unsigned __int8)byte_5057700[a2];
  if ( (*(_BYTE *)(a1 + 199) & 0x40) == 0 || !*(_BYTE *)(a1 + a2 + 216) )
  {
    if ( (_BYTE)result == 8 )
    {
      **(_QWORD **)(a1 + 8LL * a2) = a3;
      return result;
    }
LABEL_7:
    abort();
  }
  *(_QWORD *)(a1 + 8LL * a2) = a3;
  return result;
}
