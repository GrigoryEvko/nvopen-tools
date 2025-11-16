// Function: sub_3248250
// Address: 0x3248250
//
unsigned __int64 __fastcall sub_3248250(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4)
{
  unsigned __int64 result; // rax
  __int64 v5; // r8

  if ( !a4 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL))(a1);
  result = *a4;
  if ( (unsigned __int8)result <= 0x21u )
  {
    v5 = 0x200230000LL;
    if ( _bittest64(&v5, result) )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL))(a1);
  }
  return result;
}
