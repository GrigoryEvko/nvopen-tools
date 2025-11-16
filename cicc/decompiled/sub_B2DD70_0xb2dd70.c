// Function: sub_B2DD70
// Address: 0xb2dd70
//
__int64 __fastcall sub_B2DD70(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 result; // rax

  *(_DWORD *)(a1 + 132) = -1;
  v1 = sub_BD5D20(a1);
  if ( v2 > 4 && *(_DWORD *)v1 == 1836477548 && *(_BYTE *)(v1 + 4) == 46 )
  {
    *(_BYTE *)(a1 + 33) |= 0x20u;
    result = sub_B60C50(v1, v2, v2, v3);
    *(_DWORD *)(a1 + 36) = result;
  }
  else
  {
    *(_QWORD *)(a1 + 32) &= 0xFFFFDFFFuLL;
    return 4294959103LL;
  }
  return result;
}
