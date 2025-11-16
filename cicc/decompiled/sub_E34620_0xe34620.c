// Function: sub_E34620
// Address: 0xe34620
//
__int64 __fastcall sub_E34620(unsigned __int8 *a1)
{
  __int64 v1; // rdx
  __int64 result; // rax

  if ( (unsigned __int8)(*a1 - 34) > 0x33u )
    return 0;
  v1 = 0x8000000000041LL;
  if ( !_bittest64(&v1, (unsigned int)*a1 - 34) )
    return 0;
  result = sub_A73ED0((_QWORD *)a1 + 9, 6);
  if ( !(_BYTE)result )
    return sub_B49560((__int64)a1, 6);
  return result;
}
