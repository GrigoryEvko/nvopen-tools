// Function: sub_630790
// Address: 0x630790
//
__int64 __fastcall sub_630790(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 194) & 0x20) == 0 )
    return a1;
  if ( !qword_4CFDE48 )
    return a1;
  v1 = sub_881B20(qword_4CFDE48, a1, 0);
  v2 = v1;
  if ( !v1 )
    return a1;
  result = *(_QWORD *)(*(_QWORD *)v1 + 8LL);
  if ( (*(_BYTE *)(result + 194) & 0x20) != 0 )
  {
    result = sub_630790(result);
    *(_QWORD *)(*(_QWORD *)v2 + 8LL) = result;
  }
  return result;
}
