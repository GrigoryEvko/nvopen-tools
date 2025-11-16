// Function: sub_DBA6E0
// Address: 0xdba6e0
//
__int64 __fastcall sub_DBA6E0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rax

  if ( a4 != 1 )
  {
    if ( a4 == 2 )
    {
      v5 = sub_DB9E00(a1, a2);
      v6 = sub_D97B30(v5, a3, 0);
      if ( v6 )
        return *(_QWORD *)(v6 + 56);
    }
    else
    {
      if ( a4 )
        BUG();
      v8 = sub_DB9E00(a1, a2);
      v9 = sub_D97B30(v8, a3, 0);
      if ( v9 )
        return *(_QWORD *)(v9 + 40);
    }
    return sub_D970F0(a1);
  }
  v10 = sub_DB9E00(a1, a2);
  v11 = sub_D97B30(v10, a3, 0);
  if ( !v11 )
    return sub_D970F0(a1);
  return *(_QWORD *)(v11 + 48);
}
