// Function: sub_124CB30
// Address: 0x124cb30
//
__int64 __fastcall sub_124CB30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  result = sub_124C860(a1);
  if ( !(_BYTE)result || *(_DWORD *)(a3 + 148) == 1879002121 )
  {
    result = 0;
    if ( a2 )
      return *(unsigned __int8 *)(a2 + 2);
  }
  return result;
}
