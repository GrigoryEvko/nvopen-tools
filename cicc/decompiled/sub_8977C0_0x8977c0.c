// Function: sub_8977C0
// Address: 0x8977c0
//
__int64 *__fastcall sub_8977C0(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *result; // rax

  if ( word_4F06418[0] != 16 )
    return sub_6793E0(a1, (__int64 *)&qword_4D03B88, 1, 0, (__int64)a2);
  result = (__int64 *)sub_7B8B50(a1, a2, a3, a4, a5, a6);
  if ( a1 )
    *(_BYTE *)(a1 + 32) |= 0x10u;
  return result;
}
