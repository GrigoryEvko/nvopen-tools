// Function: sub_31A4B60
// Address: 0x31a4b60
//
char *__fastcall sub_31A4B60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // eax

  if ( *(_DWORD *)(a1 + 88) != 1 && *(_DWORD *)(a1 + 8) == 1 )
    return "loop-vectorize";
  v6 = *(_DWORD *)(a1 + 40);
  if ( v6 == -1 )
  {
    if ( (unsigned __int8)sub_F6E590(*(_QWORD *)(a1 + 104), a2, a3, a4, a5, a6) )
      return "loop-vectorize";
    v12 = *(_DWORD *)(a1 + 40);
    if ( !v12
      || v12 == -1
      && !(unsigned __int8)sub_F6E590(*(_QWORD *)(a1 + 104), a2, v8, v9, v10, v11)
      && *(_DWORD *)(a1 + 40) == -1
      && !*(_DWORD *)(a1 + 8) )
    {
      return "loop-vectorize";
    }
  }
  else if ( !v6 )
  {
    return "loop-vectorize";
  }
  return (char *)off_4B91160;
}
