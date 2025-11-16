// Function: sub_318D240
// Address: 0x318d240
//
__int64 __fastcall sub_318D240(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rbx
  __int64 *v4; // r14
  unsigned int v6; // r15d
  __int64 v7; // rdi

  v3 = *(__int64 **)(a1 + 40);
  v4 = &v3[*(unsigned int *)(a1 + 48)];
  if ( v4 == v3 )
  {
    return 0;
  }
  else
  {
    v6 = 0;
    do
    {
      v7 = *v3++;
      v6 |= (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v7 + 24LL))(v7, a2, a3);
    }
    while ( v4 != v3 );
  }
  return v6;
}
