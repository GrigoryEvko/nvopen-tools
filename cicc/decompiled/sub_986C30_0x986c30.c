// Function: sub_986C30
// Address: 0x986c30
//
__int64 __fastcall sub_986C30(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)result > 0x40 )
    return sub_C482E0(a1, a2);
  if ( (_DWORD)result == (_DWORD)a2 )
    *(_QWORD *)a1 = 0;
  else
    *(_QWORD *)a1 >>= a2;
  return result;
}
