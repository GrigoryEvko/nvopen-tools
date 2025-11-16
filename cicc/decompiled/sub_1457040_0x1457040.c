// Function: sub_1457040
// Address: 0x1457040
//
__int64 __fastcall sub_1457040(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rdi

  v4 = *(_QWORD **)a1;
  v5 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
  if ( v5 == *(_QWORD *)a1 )
    return sub_1456E90(a3);
  while ( 1 )
  {
    if ( a2 == *v4 )
    {
      v6 = v4[2];
      if ( !v6 || sub_1452CB0(v6) )
        break;
    }
    v4 += 3;
    if ( (_QWORD *)v5 == v4 )
      return sub_1456E90(a3);
  }
  return v4[1];
}
