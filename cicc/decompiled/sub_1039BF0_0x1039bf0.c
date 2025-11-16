// Function: sub_1039BF0
// Address: 0x1039bf0
//
__int64 __fastcall sub_1039BF0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 result; // rax

  v1 = **(_QWORD **)(a1 + 8);
  if ( *(_BYTE *)v1 != 1 || (v2 = *(_QWORD *)(v1 + 136), *(_BYTE *)v2 != 17) )
    BUG();
  result = *(_QWORD *)(v2 + 24);
  if ( *(_DWORD *)(v2 + 32) > 0x40u )
    return *(_QWORD *)result;
  return result;
}
