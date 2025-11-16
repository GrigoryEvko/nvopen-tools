// Function: sub_2D284A0
// Address: 0x2d284a0
//
__int64 __fastcall sub_2D284A0(__int64 a1)
{
  __int64 v1; // rax
  bool v2; // zf
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v2 = *(_DWORD *)(v1 + 36) == 69;
  result = 0;
  if ( v2 )
    return a1;
  return result;
}
