// Function: sub_39AC500
// Address: 0x39ac500
//
char __fastcall sub_39AC500(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rax
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 i; // r12
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 (__fastcall *v9)(__int64, __int64); // [rsp+8h] [rbp-38h]

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
  v2 = *(_QWORD *)(a1 + 16);
  v3 = *(_QWORD *)(v2 + 1688);
  v4 = *(_QWORD *)(v3 + 32);
  for ( i = v3 + 24; i != v4; v4 = *(_QWORD *)(v4 + 8) )
  {
    while ( 1 )
    {
      v6 = 0;
      if ( v4 )
        v6 = v4 - 56;
      LOBYTE(v2) = sub_15602E0((_QWORD *)(v6 + 112), "safeseh", 7u);
      if ( (_BYTE)v2 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( i == v4 )
        return v2;
    }
    v9 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v1 + 304LL);
    v7 = sub_396EAF0(*(_QWORD *)(a1 + 8), v6);
    LOBYTE(v2) = v9(v1, v7);
  }
  return v2;
}
