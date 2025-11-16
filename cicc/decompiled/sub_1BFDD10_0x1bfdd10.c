// Function: sub_1BFDD10
// Address: 0x1bfdd10
//
unsigned __int64 __fastcall sub_1BFDD10(unsigned __int64 *a1, __int64 a2)
{
  __int64 v3; // rsi
  unsigned __int64 result; // rax

  v3 = a1[1];
  if ( v3 == a1[2] )
    return sub_1BFDB20(a1, v3, a2);
  if ( v3 )
  {
    *(_QWORD *)v3 = *(_QWORD *)a2;
    result = *(unsigned __int8 *)(a2 + 16);
    *(_BYTE *)(v3 + 16) = result;
    if ( (_BYTE)result )
    {
      result = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(v3 + 8) = result;
    }
    v3 = a1[1];
  }
  a1[1] = v3 + 24;
  return result;
}
