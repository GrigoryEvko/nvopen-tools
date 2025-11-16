// Function: sub_2ECDC20
// Address: 0x2ecdc20
//
void __fastcall sub_2ECDC20(__int64 a1)
{
  _QWORD *i; // rbx
  unsigned __int64 v3; // r13
  __int64 *v4; // rdi

  if ( *(_QWORD *)(a1 + 16) > 1u )
  {
    sub_2ECD9B0(a1, (unsigned __int8 (__fastcall *)(__int64 *, _QWORD *))sub_2EC0BE0);
    for ( i = **(_QWORD ***)a1; (_QWORD *)a1 != i; i = (_QWORD *)*i )
    {
      while ( 1 )
      {
        v3 = i[1];
        if ( *(_QWORD *)(v3 + 24) >= i[2] )
          break;
        i = (_QWORD *)*i;
        if ( (_QWORD *)a1 == i )
          return;
      }
      v4 = (__int64 *)i[1];
      i[2] = *(_QWORD *)(v3 + 16);
      --*(_QWORD *)(a1 + 16);
      sub_2208CA0(v4);
      j_j___libc_free_0(v3);
    }
  }
}
