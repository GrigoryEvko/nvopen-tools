// Function: sub_5EB9D0
// Address: 0x5eb9d0
//
__int64 __fastcall sub_5EB9D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // r13
  __int64 v5; // rdi
  __int64 v6; // r12
  __int64 v7; // rax

  if ( (*(_BYTE *)(a2 + 96) & 2) != 0 )
    return 0;
  v3 = *(_QWORD *)(a2 + 112);
  if ( (*(_BYTE *)(v3 + 24) & 1) == 0 )
  {
    v4 = *(_QWORD **)(v3 + 8);
    v5 = v4[2];
    if ( a2 != v5 )
    {
      v6 = *(_QWORD *)(a1 + 56);
      do
      {
        v7 = sub_8E5310(v5, v6, a1);
        v4 = (_QWORD *)*v4;
        a1 = v7;
        v5 = v4[2];
      }
      while ( v5 != a2 );
    }
  }
  return a1;
}
