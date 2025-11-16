// Function: sub_18B1AF0
// Address: 0x18b1af0
//
_BOOL8 __fastcall sub_18B1AF0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r14
  bool v3; // al
  __int64 v4; // rbx
  __int64 v5; // r12
  bool v7; // [rsp+Fh] [rbp-31h]

  v1 = *(_QWORD *)(a1 + 32);
  v7 = 0;
  while ( a1 + 24 != v1 )
  {
    v2 = v1;
    v1 = *(_QWORD *)(v1 + 8);
    v3 = sub_15E4F60(v2 - 56);
    if ( v3 && !*(_QWORD *)(v2 - 48) )
    {
      v7 = v3;
      sub_15E3D00(v2 - 56);
    }
  }
  v4 = *(_QWORD *)(a1 + 16);
  while ( a1 + 8 != v4 )
  {
    v5 = v4;
    v4 = *(_QWORD *)(v4 + 8);
    if ( sub_15E4F60(v5 - 56) && !*(_QWORD *)(v5 - 48) )
      sub_15E55B0(v5 - 56);
  }
  return v7;
}
