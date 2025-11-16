// Function: sub_6416D0
// Address: 0x6416d0
//
__int64 __fastcall sub_6416D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // r14d
  __int64 v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // r8

  v2 = a2;
  v3 = a1;
  if ( (unsigned int)sub_8D2310(a1) )
  {
    while ( *(_BYTE *)(v3 + 140) == 12 )
      v3 = *(_QWORD *)(v3 + 160);
    v4 = *(_QWORD *)(v3 + 160);
    if ( *(_BYTE *)(a2 + 140) == 12 )
    {
      do
        v2 = *(_QWORD *)(v2 + 160);
      while ( *(_BYTE *)(v2 + 140) == 12 );
    }
    v5 = *(_QWORD *)(v2 + 160);
    if ( v4 == v5
      || (v6 = sub_8DED30(*(_QWORD *)(v3 + 160), *(_QWORD *)(v2 + 160), 1)) != 0
      || (unsigned int)sub_8D2930(v4) && (unsigned int)sub_8DED40(v4, v5) )
    {
      v6 = 1;
      if ( ((*(_BYTE *)(*(_QWORD *)(v2 + 168) + 16LL) ^ *(_BYTE *)(*(_QWORD *)(v3 + 168) + 16LL)) & 2) == 0 )
      {
        *(_QWORD *)(v3 + 160) = v5;
        if ( v2 != v3 )
          v6 = sub_8DED30(v3, v2, 1) != 0;
        *(_QWORD *)(v3 + 160) = v4;
      }
    }
    return v6;
  }
  if ( (unsigned int)sub_8D3410(a1) )
  {
    v6 = 1;
    v8 = sub_8D4050(a1);
    v10 = sub_8D4050(a2);
    if ( v8 != v10 )
      return (unsigned int)sub_8D97D0(v8, v10, 0, v9, v11) != 0;
    return v6;
  }
  return sub_8DED40(a1, a2);
}
