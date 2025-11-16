// Function: sub_B2B840
// Address: 0xb2b840
//
__int64 __fastcall sub_B2B840(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v6; // r14
  __int64 v7; // r12
  bool v8; // bl
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // [rsp+0h] [rbp-40h]

  result = a2 - 72;
  v5 = a1 - 72;
  if ( a1 - 72 != a2 - 72 )
  {
    result = *(_QWORD *)(a2 + 40);
    v6 = *(_QWORD *)(a1 + 40);
    v7 = a3;
    v11 = result;
    if ( v6 == result )
    {
      if ( a3 != a4 )
      {
        do
        {
          v10 = v7 - 24;
          if ( !v7 )
            v10 = 0;
          result = sub_AA64B0(v10, v5);
          v7 = *(_QWORD *)(v7 + 8);
        }
        while ( v7 != a4 );
      }
    }
    else if ( a3 != a4 )
    {
      do
      {
        if ( !v7 )
          BUG();
        v8 = (*(_BYTE *)(v7 - 17) & 0x10) != 0;
        if ( v11 && (*(_BYTE *)(v7 - 17) & 0x10) != 0 )
        {
          v9 = sub_BD5C70(v7 - 24);
          sub_BD8AE0(v11, v9);
        }
        result = sub_AA64B0(v7 - 24, v5);
        if ( v6 )
        {
          if ( v8 )
            result = sub_BD8920(v6, v7 - 24);
        }
        v7 = *(_QWORD *)(v7 + 8);
      }
      while ( v7 != a4 );
    }
  }
  return result;
}
