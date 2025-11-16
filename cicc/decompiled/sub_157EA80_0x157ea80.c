// Function: sub_157EA80
// Address: 0x157ea80
//
__int64 __fastcall sub_157EA80(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 result; // rax
  bool v9; // bl
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // [rsp+0h] [rbp-40h]

  v6 = a1 - 40;
  v7 = sub_157E9B0(a1 - 40);
  result = sub_157E9B0(a2 - 40);
  v12 = result;
  if ( result == v7 )
  {
    for ( ; a3 != a4; a3 = *(_QWORD *)(a3 + 8) )
    {
      v11 = a3 - 24;
      if ( !a3 )
        v11 = 0;
      result = sub_15F2040(v11, v6);
    }
  }
  else
  {
    for ( ; a3 != a4; a3 = *(_QWORD *)(a3 + 8) )
    {
      if ( !a3 )
        BUG();
      v9 = (*(_BYTE *)(a3 - 1) & 0x20) != 0;
      if ( v12 && (*(_BYTE *)(a3 - 1) & 0x20) != 0 )
      {
        v10 = sub_16498B0(a3 - 24);
        sub_164D860(v12, v10);
      }
      result = sub_15F2040(a3 - 24, v6);
      if ( v7 )
      {
        if ( v9 )
          result = sub_164D6D0(v7, a3 - 24);
      }
    }
  }
  return result;
}
