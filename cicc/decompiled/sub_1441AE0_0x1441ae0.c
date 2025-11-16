// Function: sub_1441AE0
// Address: 0x1441ae0
//
__int64 __fastcall sub_1441AE0(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rdi

  if ( a1[1] )
    return 1;
  v3 = sub_1633D90(*a1);
  result = 0;
  if ( v3 )
  {
    v4 = sub_163B090();
    v5 = a1[1];
    a1[1] = v4;
    if ( v5 )
    {
      v6 = *(_QWORD *)(v5 + 8);
      if ( v6 )
        j_j___libc_free_0(v6, *(_QWORD *)(v5 + 24) - v6);
      j_j___libc_free_0(v5, 72);
    }
    return 1;
  }
  return result;
}
