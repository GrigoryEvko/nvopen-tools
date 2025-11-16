// Function: sub_E13A60
// Address: 0xe13a60
//
__int64 *__fastcall sub_E13A60(__int64 a1, __int64 *a2)
{
  __int64 *result; // rax

  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 32LL))(*(_QWORD *)(a1 + 16));
  result = (__int64 *)*(unsigned int *)(a1 + 12);
  if ( ((unsigned __int8)result & 1) != 0 )
  {
    sub_E12F20(a2, 6u, " const");
    result = (__int64 *)*(unsigned int *)(a1 + 12);
    if ( ((unsigned __int8)result & 2) == 0 )
    {
LABEL_3:
      if ( ((unsigned __int8)result & 4) == 0 )
        return result;
      return sub_E12F20(a2, 9u, " restrict");
    }
  }
  else if ( ((unsigned __int8)result & 2) == 0 )
  {
    goto LABEL_3;
  }
  sub_E12F20(a2, 9u, " volatile");
  result = (__int64 *)*(unsigned int *)(a1 + 12);
  if ( ((unsigned __int8)result & 4) != 0 )
    return sub_E12F20(a2, 9u, " restrict");
  return result;
}
