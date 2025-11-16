// Function: sub_987160
// Address: 0x987160
//
unsigned __int64 __fastcall sub_987160(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  unsigned __int64 result; // rax

  v5 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v5 > 0x40 )
    return sub_C43D10(a1, a2, v5, a4, a5);
  result = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~*(_QWORD *)a1;
  if ( !(_DWORD)v5 )
    result = 0;
  *(_QWORD *)a1 = result;
  return result;
}
