// Function: sub_B96E90
// Address: 0xb96e90
//
__int64 __fastcall sub_B96E90(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  __int64 result; // rax

  v4 = sub_B91060((unsigned __int8 *)a2);
  if ( v4 )
  {
    sub_B96CB0(v4, a1, a3);
    return 1;
  }
  else
  {
    result = 0;
    if ( *(_BYTE *)a2 == 3 )
    {
      *(_QWORD *)(a2 + 8) = a1;
      return 1;
    }
  }
  return result;
}
