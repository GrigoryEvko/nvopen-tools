// Function: sub_33C94B0
// Address: 0x33c94b0
//
__int64 __fastcall sub_33C94B0(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  __int64 result; // rax
  unsigned __int64 v4; // rsi
  unsigned __int64 v5; // rsi

  v2 = *(_DWORD *)(a1 + 8);
  result = a2;
  v4 = *(_QWORD *)a1;
  if ( v2 <= 0x40 )
  {
    v5 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & (result | v4);
    result = 0;
    if ( !v2 )
      v5 = 0;
    *(_QWORD *)a1 = v5;
  }
  else
  {
    *(_QWORD *)v4 |= result;
  }
  return result;
}
