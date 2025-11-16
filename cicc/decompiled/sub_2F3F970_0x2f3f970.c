// Function: sub_2F3F970
// Address: 0x2f3f970
//
const char *__fastcall sub_2F3F970(__int64 a1)
{
  int v1; // eax

  v1 = *(_DWORD *)(a1 + 184);
  if ( v1 == 1 )
    return "Release mode Regalloc Eviction Advisor";
  if ( v1 == 2 )
    return "Development mode Regalloc Eviction Advisor";
  if ( v1 )
    BUG();
  return "Default Regalloc Eviction Advisor";
}
