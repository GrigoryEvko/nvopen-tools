// Function: sub_1BC2C00
// Address: 0x1bc2c00
//
__int64 __fastcall sub_1BC2C00(__int64 a1)
{
  __int64 v1; // rcx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 24);
  for ( result = *(_QWORD *)(a1 + 16); result != v1; *(_QWORD *)(a1 + 16) = result )
  {
    if ( *(_DWORD *)(result + 8) != 1 || **(_DWORD **)result != -2 && **(_DWORD **)result != -3 )
      break;
    result += 40;
  }
  return result;
}
