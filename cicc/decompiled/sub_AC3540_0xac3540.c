// Function: sub_AC3540
// Address: 0xac3540
//
__int64 __fastcall sub_AC3540(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // r13

  v1 = *a1;
  v2 = *(_QWORD *)(*a1 + 2632);
  if ( !v2 )
  {
    v2 = sub_BD2C40(24, unk_3F289A4);
    if ( v2 )
    {
      v4 = sub_BCB190(a1);
      sub_BD35F0(v2, v4, 21);
      *(_DWORD *)(v2 + 4) &= 0x38000000u;
    }
    v5 = *(_QWORD *)(v1 + 2632);
    *(_QWORD *)(v1 + 2632) = v2;
    if ( v5 )
    {
      sub_BD7260(v5);
      sub_BD2DD0(v5);
      return *(_QWORD *)(v1 + 2632);
    }
  }
  return v2;
}
