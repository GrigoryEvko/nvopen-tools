// Function: sub_15809C0
// Address: 0x15809c0
//
__int64 __fastcall sub_15809C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 result; // rax
  __int64 v7; // r13
  bool v8; // bl
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // [rsp+0h] [rbp-40h]

  v4 = a1 - 72;
  v5 = a3;
  result = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)(a1 + 32);
  v11 = result;
  if ( result == v7 )
  {
    if ( a3 != a4 )
    {
      do
      {
        v10 = v5 - 24;
        if ( !v5 )
          v10 = 0;
        result = sub_157F970(v10, v4);
        v5 = *(_QWORD *)(v5 + 8);
      }
      while ( v5 != a4 );
    }
  }
  else if ( a3 != a4 )
  {
    do
    {
      if ( !v5 )
        BUG();
      v8 = (*(_BYTE *)(v5 - 1) & 0x20) != 0;
      if ( v11 && (*(_BYTE *)(v5 - 1) & 0x20) != 0 )
      {
        v9 = sub_16498B0(v5 - 24);
        sub_164D860(v11, v9);
      }
      result = sub_157F970(v5 - 24, v4);
      if ( v7 )
      {
        if ( v8 )
          result = sub_164D6D0(v7, v5 - 24);
      }
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v5 != a4 );
  }
  return result;
}
