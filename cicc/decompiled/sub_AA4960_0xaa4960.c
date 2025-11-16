// Function: sub_AA4960
// Address: 0xaa4960
//
void __fastcall sub_AA4960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v7; // r13
  __int64 v8; // r14
  bool v9; // r12
  __int64 v10; // rax

  v5 = a1 - 48;
  sub_AA48B0(a1 - 48);
  if ( a1 - 48 != a2 - 48 )
  {
    v7 = sub_AA4890(v5);
    v8 = sub_AA4890(a2 - 48);
    if ( v7 == v8 )
    {
      for ( ; a5 != a3; a3 = *(_QWORD *)(a3 + 8) )
      {
        if ( !a3 )
        {
          MEMORY[0x28] = a1 - 48;
          BUG();
        }
        *(_QWORD *)(a3 + 16) = v5;
      }
    }
    else
    {
      for ( ; a5 != a3; a3 = *(_QWORD *)(a3 + 8) )
      {
        if ( !a3 )
          BUG();
        v9 = (*(_BYTE *)(a3 - 17) & 0x10) != 0;
        if ( v8 && (*(_BYTE *)(a3 - 17) & 0x10) != 0 )
        {
          v10 = sub_BD5C70(a3 - 24);
          sub_BD8AE0(v8, v10);
        }
        *(_QWORD *)(a3 + 16) = v5;
        if ( v7 )
        {
          if ( v9 )
            sub_BD8920(v7, a3 - 24);
        }
      }
    }
  }
}
