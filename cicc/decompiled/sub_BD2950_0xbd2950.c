// Function: sub_BD2950
// Address: 0xbd2950
//
void __fastcall sub_BD2950(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rax

  while ( a1 != a2 )
  {
    a2 -= 32;
    if ( *(_QWORD *)a2 )
    {
      v3 = *(_QWORD *)(a2 + 8);
      **(_QWORD **)(a2 + 16) = v3;
      if ( v3 )
        *(_QWORD *)(v3 + 16) = *(_QWORD *)(a2 + 16);
    }
  }
  if ( a3 )
    j___libc_free_0(a1);
}
