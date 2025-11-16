// Function: sub_E45340
// Address: 0xe45340
//
__int64 __fastcall sub_E45340(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = 0;
  if ( *(_BYTE *)a2 > 0x1Cu )
    return *(_QWORD *)(a2 + 40);
  return result;
}
