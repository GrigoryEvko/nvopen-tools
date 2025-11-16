// Function: sub_204D410
// Address: 0x204d410
//
__int64 __fastcall sub_204D410(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax
  __int64 v4; // rsi

  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = a3;
  if ( a2 )
  {
    result = a2 + 48;
    if ( a1 != a2 + 48 )
    {
      v4 = *(_QWORD *)(a2 + 48);
      *(_QWORD *)a1 = v4;
      if ( v4 )
        return sub_1623A60(a1, v4, 2);
    }
  }
  return result;
}
