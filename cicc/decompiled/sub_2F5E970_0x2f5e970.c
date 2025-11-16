// Function: sub_2F5E970
// Address: 0x2f5e970
//
const char *__fastcall sub_2F5E970(__int64 a1)
{
  int v1; // eax

  v1 = *(_DWORD *)(a1 + 184);
  if ( v1 != 2 )
  {
    if ( v1 > 2 )
    {
      if ( v1 == 3 )
        return "Dummy Regalloc Priority Advisor";
    }
    else
    {
      if ( !v1 )
        return "Default Regalloc Priority Advisor";
      if ( v1 == 1 )
        return "Release mode Regalloc Priority Advisor";
    }
    BUG();
  }
  return "Development mode Regalloc Priority Advisor";
}
