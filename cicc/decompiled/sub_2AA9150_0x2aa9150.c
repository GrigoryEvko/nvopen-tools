// Function: sub_2AA9150
// Address: 0x2aa9150
//
unsigned __int64 __fastcall sub_2AA9150(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // rdx
  unsigned __int64 result; // rax

  if ( *(_DWORD *)(a2 + 8) == 1 )
    *(_DWORD *)(a1 + 8) = 1;
  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)a2;
  result = *(_QWORD *)a2 * *(_QWORD *)a1;
  if ( is_mul_ok(*(_QWORD *)a2, *(_QWORD *)a1) )
    goto LABEL_4;
  if ( v2 <= 0 )
  {
    if ( v3 < 0 )
    {
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v2 < 0 )
      {
LABEL_4:
        *(_QWORD *)a1 = result;
        return result;
      }
    }
    *(_QWORD *)a1 = 0x8000000000000000LL;
    return 0x8000000000000000LL;
  }
  else
  {
    result = 0x8000000000000000LL;
    if ( v3 > 0 )
      result = 0x7FFFFFFFFFFFFFFFLL;
    *(_QWORD *)a1 = result;
  }
  return result;
}
