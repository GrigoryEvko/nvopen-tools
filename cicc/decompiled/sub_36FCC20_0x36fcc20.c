// Function: sub_36FCC20
// Address: 0x36fcc20
//
char __fastcall sub_36FCC20(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  _QWORD *i; // rbx
  __int64 v4; // r12

  v2 = a2[22];
  *(_QWORD *)(a1 + 56) = a2[21];
  *(_QWORD *)(a1 + 64) = v2;
  for ( i = (_QWORD *)a2[4]; a2 + 3 != i; i = (_QWORD *)i[1] )
  {
    v4 = (__int64)(i - 7);
    if ( !i )
      v4 = 0;
    LOBYTE(v2) = sub_B2FC80(v4);
    if ( !(_BYTE)v2 )
    {
      ++*(_DWORD *)(a1 + 48);
      v2 = sub_B91CC0(v4, "thinlto_src_module", 0x12u);
      *(_DWORD *)(a1 + 52) -= (v2 == 0) - 1;
    }
  }
  return v2;
}
