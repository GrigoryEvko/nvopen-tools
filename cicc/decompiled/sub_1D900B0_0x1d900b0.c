// Function: sub_1D900B0
// Address: 0x1d900b0
//
__int64 __fastcall sub_1D900B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 i; // r13
  __int64 v6; // r12

  v2 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FC3606, 1u);
  v3 = v2;
  if ( v2 )
    v3 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_4FC3606);
  v4 = *(_QWORD *)(a2 + 32);
  for ( i = a2 + 24; i != v4; v4 = *(_QWORD *)(v4 + 8) )
  {
    while ( 1 )
    {
      v6 = v4 - 56;
      if ( !v4 )
        v6 = 0;
      if ( !sub_15E4F60(v6) && (*(_BYTE *)(v6 + 19) & 0x40) != 0 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( i == v4 )
        return 0;
    }
    sub_1D8F610(v3, v6);
  }
  return 0;
}
