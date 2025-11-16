// Function: sub_3703310
// Address: 0x3703310
//
_WORD *__fastcall sub_3703310(__int64 a1, unsigned int a2, int a3, __int64 a4)
{
  unsigned __int64 v4; // r8
  _WORD *result; // rax
  __int64 v6; // rdx

  v4 = a3 - a2;
  result = (_WORD *)(*(_QWORD *)(a1 + 48) + a2);
  *result = a3 - a2 - 2;
  if ( BYTE4(a4) )
  {
    v6 = (__int64)result + v4 - 8;
    if ( v4 <= 8 )
      v6 = (__int64)result;
    *(_DWORD *)(v6 + 4) = a4;
  }
  return result;
}
