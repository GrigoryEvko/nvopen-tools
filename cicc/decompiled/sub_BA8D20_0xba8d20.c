// Function: sub_BA8D20
// Address: 0xba8d20
//
_BYTE *__fastcall sub_BA8D20(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 (__fastcall *a5)(__int64),
        __int64 a6)
{
  _BYTE *result; // rax

  result = (_BYTE *)sub_BA8B30(a1, a2, a3);
  if ( !result || *result != 3 )
    return (_BYTE *)a5(a6);
  return result;
}
