// Function: sub_1EA3220
// Address: 0x1ea3220
//
__int64 __fastcall sub_1EA3220(int a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v5; // rdx

  if ( a1 < 0 )
    v2 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 16LL * (a1 & 0x7FFFFFFF) + 8);
  else
    v2 = *(_QWORD *)(*(_QWORD *)(a2 + 272) + 8LL * (unsigned int)a1);
  if ( !v2 )
    return 1;
  if ( (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
  {
    v2 = *(_QWORD *)(v2 + 32);
    if ( !v2 || (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
      return 1;
  }
  v3 = *(_QWORD *)(v2 + 16);
  if ( **(_WORD **)(v3 + 16) == 9 )
  {
    while ( 1 )
    {
      v2 = *(_QWORD *)(v2 + 32);
      if ( !v2 || (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
        break;
      v5 = *(_QWORD *)(v2 + 16);
      if ( v5 != v3 )
      {
        v3 = *(_QWORD *)(v2 + 16);
        if ( **(_WORD **)(v5 + 16) != 9 )
          return 0;
      }
    }
    return 1;
  }
  return 0;
}
