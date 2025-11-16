// Function: sub_DBAF70
// Address: 0xdbaf70
//
__int64 __fastcall sub_DBAF70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 *v7; // rax
  __int64 v8; // rax
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 v13; // rax

  if ( a5 != 1 )
  {
    if ( a5 == 2 )
    {
      v7 = sub_DBAF30(a1, a2);
      v8 = sub_D97B30(v7, a3, a4);
      if ( v8 )
        return *(_QWORD *)(v8 + 56);
    }
    else
    {
      if ( a5 )
        BUG();
      v10 = sub_DBAF30(a1, a2);
      v11 = sub_D97B30(v10, a3, a4);
      if ( v11 )
        return *(_QWORD *)(v11 + 40);
    }
    return sub_D970F0(a1);
  }
  v12 = sub_DBAF30(a1, a2);
  v13 = sub_D97B30(v12, a3, a4);
  if ( !v13 )
    return sub_D970F0(a1);
  return *(_QWORD *)(v13 + 48);
}
