// Function: sub_36D00D0
// Address: 0x36d00d0
//
__int64 __fastcall sub_36D00D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rcx
  __int64 v4; // r15
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // r12
  __int64 result; // rax
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 i; // r14
  __int64 v11; // rdi

  v3 = *(_QWORD *)(a2 + 80);
  v4 = *(_QWORD *)(v3 + 8);
  v5 = *(_QWORD *)(v3 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 == v3 + 24 )
  {
    v6 = 0;
  }
  else
  {
    if ( !v5 )
      BUG();
    v6 = v5 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 >= 0xB )
      v6 = 0;
  }
  result = 0;
  if ( v4 != a2 + 72 )
  {
    v8 = v6 + 24;
    do
    {
      if ( !v4 )
        BUG();
      v9 = *(_QWORD *)(v4 + 32);
      for ( i = v4 + 24; i != v9; result = 1 )
      {
        while ( 1 )
        {
          v11 = v9;
          v9 = *(_QWORD *)(v9 + 8);
          if ( *(_BYTE *)(v11 - 24) == 60 && **(_BYTE **)(v11 - 56) == 17 )
            break;
          if ( i == v9 )
            goto LABEL_14;
        }
        LOWORD(v2) = 0;
        sub_B444E0((_QWORD *)(v11 - 24), v8, v2);
      }
LABEL_14:
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( a2 + 72 != v4 );
  }
  return result;
}
