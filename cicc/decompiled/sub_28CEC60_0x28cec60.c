// Function: sub_28CEC60
// Address: 0x28cec60
//
unsigned __int64 __fastcall sub_28CEC60(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdx
  __int64 v5; // rsi
  unsigned __int64 result; // rax

  v4 = *a2;
  v5 = 4LL * *(unsigned int *)(a1 + 48);
  a2[10] += v5;
  result = (v4 + 3) & 0xFFFFFFFFFFFFFFFCLL;
  if ( a2[1] >= v5 + result && v4 )
  {
    *a2 = v5 + result;
    *(_QWORD *)(a1 + 56) = result;
  }
  else
  {
    result = sub_9D1E70((__int64)a2, v5, v5, 2);
    *(_QWORD *)(a1 + 56) = result;
  }
  return result;
}
