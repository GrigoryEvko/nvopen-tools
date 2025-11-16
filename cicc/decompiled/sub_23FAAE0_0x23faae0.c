// Function: sub_23FAAE0
// Address: 0x23faae0
//
__int64 __fastcall sub_23FAAE0(unsigned __int8 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  if ( (unsigned __int8)(*a1 - 42) > 0x34u )
    return 0;
  v3 = 0x1F133FFE23FFFFLL;
  if ( !_bittest64(&v3, (unsigned int)*a1 - 42) )
    return 0;
  LOBYTE(result) = sub_991A70(a1, 0, 0, a2, 0, 1u, 0);
  return (unsigned int)result;
}
