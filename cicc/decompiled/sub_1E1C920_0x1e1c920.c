// Function: sub_1E1C920
// Address: 0x1e1c920
//
__int64 __fastcall sub_1E1C920(__int64 a1)
{
  __int64 v1; // r12
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax

  v1 = *(_QWORD *)(a1 + 616);
  if ( v1 == -1 )
    return 0;
  if ( !v1 )
  {
    v4 = sub_1E29990(*(_QWORD *)(a1 + 608));
    *(_QWORD *)(a1 + 616) = v4;
    v1 = v4;
    if ( !v4 )
    {
      v5 = sub_1E29910(*(_QWORD *)(a1 + 608));
      if ( v5
        && (v6 = sub_1DD9FB0(v5, **(_QWORD **)(*(_QWORD *)(a1 + 608) + 32LL), a1), (*(_QWORD *)(a1 + 616) = v6) != 0) )
      {
        return v6;
      }
      else
      {
        *(_QWORD *)(a1 + 616) = -1;
      }
    }
  }
  return v1;
}
