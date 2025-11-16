// Function: sub_1993E40
// Address: 0x1993e40
//
__int64 __fastcall sub_1993E40(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v3; // rdx
  __int64 *v4; // r14
  __int64 *v5; // rbx
  __int64 v6; // rax
  unsigned int v7; // eax
  unsigned int v8; // r13d
  __int64 v9; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]

  v2 = sub_157F280(**(_QWORD **)(*(_QWORD *)(a1 + 48) + 32LL));
  if ( v3 == (__int64 *)v2 )
  {
    return 0;
  }
  else
  {
    v4 = v3;
    v5 = (__int64 *)v2;
    while ( 1 )
    {
      LOBYTE(v7) = sub_1456C80(a2, *v5);
      v8 = v7;
      if ( (_BYTE)v7 )
      {
        v11 = sub_1456E10(a2, *v5);
        v9 = sub_1456040(**(_QWORD **)(a1 + 32));
        if ( v11 == sub_1456E10(a2, v9) && a1 == sub_146F1B0(a2, (__int64)v5) )
          break;
      }
      v6 = v5[4];
      if ( !v6 )
        BUG();
      v5 = 0;
      if ( *(_BYTE *)(v6 - 8) == 77 )
        v5 = (__int64 *)(v6 - 24);
      if ( v4 == v5 )
        return 0;
    }
  }
  return v8;
}
