// Function: sub_2C1B6B0
// Address: 0x2c1b6b0
//
__int64 __fastcall sub_2C1B6B0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi

  v2 = *(_QWORD *)(a1 + 80);
  if ( *(_DWORD *)(v2 + 64) )
  {
    v3 = *(_QWORD *)(*(_QWORD *)(v2 + 56) + 8LL * a2);
    return sub_2BF0520(v3);
  }
  v3 = *(_QWORD *)(v2 + 48);
  if ( a2 )
    return sub_2BF0520(v3);
  if ( *(_DWORD *)(v3 + 64) == 1 )
    return sub_2BF0520(**(_QWORD **)(v3 + 56));
  else
    return sub_2BF0520(0);
}
