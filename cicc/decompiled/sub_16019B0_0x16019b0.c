// Function: sub_16019B0
// Address: 0x16019b0
//
__int64 __fastcall sub_16019B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_1648A60(56, *(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v2 = v1;
  if ( v1 )
    sub_15F80C0(v1, a1);
  return v2;
}
