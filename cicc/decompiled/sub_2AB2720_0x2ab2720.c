// Function: sub_2AB2720
// Address: 0x2ab2720
//
__int64 __fastcall sub_2AB2720(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  __int64 v4; // rax
  __int64 *v5; // r13

  result = **(_QWORD **)(a1 + 8);
  if ( !result )
  {
    v4 = sub_BCD140(*(_QWORD **)(a2 + 72), a3);
    v5 = *(__int64 **)(a1 + 8);
    *v5 = sub_2AB2710(a2, v4, *(_QWORD *)a1);
    return **(_QWORD **)(a1 + 8);
  }
  return result;
}
