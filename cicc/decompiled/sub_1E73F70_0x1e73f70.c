// Function: sub_1E73F70
// Address: 0x1e73f70
//
__int64 __fastcall sub_1E73F70(__int64 a1, char a2)
{
  __int64 v2; // rdx
  __int64 result; // rax
  __int64 v4; // rdx
  bool v5; // cl

  v2 = *(_QWORD *)(a1 + 8);
  result = 0;
  if ( **(_WORD **)(v2 + 16) == 15 )
  {
    v4 = *(_QWORD *)(v2 + 32);
    if ( a2 )
    {
      result = 1;
      if ( *(int *)(v4 + 48) > 0 )
        return result;
      v5 = *(_DWORD *)(a1 + 212) == 0;
    }
    else
    {
      result = 1;
      if ( *(int *)(v4 + 8) > 0 )
        return result;
      v5 = *(_DWORD *)(a1 + 208) == 0;
      v4 += 40;
    }
    result = 0;
    if ( *(int *)(v4 + 8) > 0 )
      return !v5 ? 1 : -1;
  }
  return result;
}
