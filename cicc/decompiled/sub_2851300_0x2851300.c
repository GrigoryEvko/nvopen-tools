// Function: sub_2851300
// Address: 0x2851300
//
__int64 __fastcall sub_2851300(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned int v7; // eax
  unsigned int v8; // r13d
  __int64 v9; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]

  v2 = sub_AA5930(**(_QWORD **)(*(_QWORD *)(a1 + 48) + 32LL));
  if ( v2 == v3 )
  {
    return 0;
  }
  else
  {
    v4 = v3;
    v5 = v2;
    while ( 1 )
    {
      LOBYTE(v7) = sub_D97040(a2, *(_QWORD *)(v5 + 8));
      v8 = v7;
      if ( (_BYTE)v7 )
      {
        v11 = sub_D97090(a2, *(_QWORD *)(v5 + 8));
        v9 = sub_D95540(**(_QWORD **)(a1 + 32));
        if ( v11 == sub_D97090(a2, v9) && (__int64 *)a1 == sub_DD8400(a2, v5) )
          break;
      }
      v6 = *(_QWORD *)(v5 + 32);
      if ( !v6 )
        BUG();
      v5 = 0;
      if ( *(_BYTE *)(v6 - 24) == 84 )
        v5 = v6 - 24;
      if ( v4 == v5 )
        return 0;
    }
  }
  return v8;
}
