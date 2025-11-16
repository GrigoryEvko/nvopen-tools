// Function: sub_728280
// Address: 0x728280
//
__int64 __fastcall sub_728280(_QWORD *a1)
{
  __int64 v2; // r12
  __int64 i; // rax
  bool v4; // cf
  __int64 v5; // rbx
  int v6; // r14d
  _QWORD *j; // rbx
  __int64 v8; // rax
  __int64 v10; // rdi

  v2 = 0;
  for ( i = a1[14]; i; v2 -= v4 - 1LL )
  {
    v4 = *(_BYTE *)(i + 88) < 0x80u;
    i = *(_QWORD *)(i + 112);
  }
  v5 = a1[13];
  if ( v5 )
  {
    v6 = dword_4F077C4;
    do
    {
      v2 -= (*(_BYTE *)(v5 + 88) < 0x80u) - 1LL;
      if ( v6 == 2 && (unsigned __int8)(*(_BYTE *)(v5 + 140) - 9) <= 2u )
      {
        v10 = *(_QWORD *)(*(_QWORD *)(v5 + 168) + 152LL);
        if ( v10 )
        {
          if ( (*(_BYTE *)(v10 + 29) & 0x20) == 0 )
            v2 += ((__int64 (*)(void))sub_728280)();
        }
      }
      v5 = *(_QWORD *)(v5 + 112);
    }
    while ( v5 );
  }
  for ( j = (_QWORD *)a1[20]; j; v2 += v8 )
  {
    v8 = sub_728280(j);
    j = (_QWORD *)*j;
  }
  return v2;
}
