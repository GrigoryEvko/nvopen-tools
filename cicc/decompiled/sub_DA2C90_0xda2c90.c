// Function: sub_DA2C90
// Address: 0xda2c90
//
__int64 __fastcall sub_DA2C90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  unsigned int v4; // r14d
  _QWORD *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9

  if ( !*(_WORD *)(a3 + 24) )
  {
    v3 = *(_QWORD *)(a3 + 32);
    v4 = *(_DWORD *)(v3 + 32);
    if ( v4 <= 0x40 )
    {
      if ( *(_QWORD *)(v3 + 24) )
        goto LABEL_4;
    }
    else if ( v4 != (unsigned int)sub_C444A0(v3 + 24) )
    {
LABEL_4:
      v5 = sub_DA2C50(a2, *(_QWORD *)(v3 + 8), 0, 0);
      sub_D97F80(a1, (__int64)v5, v6, v7, v8, v9);
      return a1;
    }
  }
  v11 = sub_D970F0(a2);
  sub_D97F80(a1, v11, v12, v13, v14, v15);
  return a1;
}
