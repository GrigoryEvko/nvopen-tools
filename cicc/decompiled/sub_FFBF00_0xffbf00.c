// Function: sub_FFBF00
// Address: 0xffbf00
//
_QWORD *__fastcall sub_FFBF00(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  _QWORD *result; // rax

  sub_FFBE30(a1, (__int64)a2);
  if ( *(_BYTE *)(a1 + 560) != 1 )
  {
    sub_FFB730(a1, (__int64)a2);
    return (_QWORD *)sub_AA5450(a2);
  }
  if ( !*(_BYTE *)(a1 + 596) )
    return sub_C8CC70(a1 + 568, (__int64)a2, (__int64)v2, v3, v4, v5);
  result = *(_QWORD **)(a1 + 576);
  v3 = *(unsigned int *)(a1 + 588);
  v2 = &result[v3];
  if ( result == v2 )
  {
LABEL_9:
    if ( (unsigned int)v3 >= *(_DWORD *)(a1 + 584) )
      return sub_C8CC70(a1 + 568, (__int64)a2, (__int64)v2, v3, v4, v5);
    *(_DWORD *)(a1 + 588) = v3 + 1;
    *v2 = a2;
    ++*(_QWORD *)(a1 + 568);
  }
  else
  {
    while ( a2 != (_QWORD *)*result )
    {
      if ( v2 == ++result )
        goto LABEL_9;
    }
  }
  return result;
}
