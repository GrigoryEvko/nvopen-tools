// Function: sub_6E84C0
// Address: 0x6e84c0
//
__int64 __fastcall sub_6E84C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_BYTE *)(a1 + 58) |= 0x10u;
  *(_QWORD *)(a1 + 8) = sub_8D46C0(a2);
  result = sub_8D3110(a2);
  if ( (_DWORD)result )
    *(_BYTE *)(a1 + 58) |= 0x20u;
  return result;
}
