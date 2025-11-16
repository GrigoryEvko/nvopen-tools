// Function: sub_FFE530
// Address: 0xffe530
//
bool __fastcall sub_FFE530(__int64 a1)
{
  __int64 v2; // rdi
  bool result; // al
  unsigned __int8 *v4; // r12

  v2 = **(_QWORD **)(a1 + 8);
  result = 0;
  if ( *(_QWORD *)a1 == *(_QWORD *)(v2 + 8) )
  {
    v4 = sub_98ACB0((unsigned __int8 *)v2, 6u);
    return v4 == sub_98ACB0(*(unsigned __int8 **)(a1 + 16), 6u);
  }
  return result;
}
