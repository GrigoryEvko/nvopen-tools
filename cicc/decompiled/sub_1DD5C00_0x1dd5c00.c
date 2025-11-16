// Function: sub_1DD5C00
// Address: 0x1dd5c00
//
__int64 __fastcall sub_1DD5C00(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax

  while ( a4 != a3 )
  {
    result = *a1;
    *(_QWORD *)(a3 + 24) = *a1;
    a3 = *(_QWORD *)(a3 + 8);
  }
  return result;
}
