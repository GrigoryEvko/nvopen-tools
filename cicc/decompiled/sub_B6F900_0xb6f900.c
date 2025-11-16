// Function: sub_B6F900
// Address: 0xb6f900
//
__int64 __fastcall sub_B6F900(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( !*(_BYTE *)(*a1 + 1656) )
  {
    *(_QWORD *)(result + 1624) = 0;
    *(_QWORD *)(result + 1632) = 0;
    *(_QWORD *)(result + 1640) = 0;
    *(_DWORD *)(result + 1648) = 0;
    *(_BYTE *)(result + 1656) = 1;
  }
  return result;
}
