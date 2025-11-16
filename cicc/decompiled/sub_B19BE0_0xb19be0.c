// Function: sub_B19BE0
// Address: 0xb19be0
//
void __fastcall sub_B19BE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax

  v3 = *(_QWORD *)(a3 + 24);
  if ( *(_BYTE *)v3 == 84 )
    sub_B19720(
      a1,
      a2,
      *(_QWORD *)(*(_QWORD *)(v3 - 8)
                + 32LL * *(unsigned int *)(v3 + 72)
                + 8LL * (unsigned int)((a3 - *(_QWORD *)(v3 - 8)) >> 5)));
  else
    sub_B196A0(a1, a2, *(_QWORD *)(v3 + 40));
}
