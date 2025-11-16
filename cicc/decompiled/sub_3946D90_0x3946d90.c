// Function: sub_3946D90
// Address: 0x3946d90
//
__int64 __fastcall sub_3946D90(
        __int64 **a1,
        unsigned __int8 *a2,
        size_t a3,
        unsigned __int8 *a4,
        size_t a5,
        __int64 a6,
        unsigned __int8 *a7,
        size_t a8,
        unsigned __int8 *a9,
        size_t a10)
{
  __int64 *v10; // r13
  __int64 *v12; // r15
  __int64 result; // rax

  v10 = a1[1];
  if ( *a1 == v10 )
    return 0;
  v12 = *a1;
  while ( 1 )
  {
    if ( (unsigned int)sub_3946C00(*v12, a2, a3) )
    {
      result = sub_3946CB0((__int64)a1, (__int64)(v12 + 1), a4, a5, a7, a8, a9, a10);
      if ( (_DWORD)result )
        break;
    }
    v12 += 5;
    if ( v10 == v12 )
      return 0;
  }
  return result;
}
