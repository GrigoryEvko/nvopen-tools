// Function: sub_829CB0
// Address: 0x829cb0
//
__int64 __fastcall sub_829CB0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rbx
  _QWORD *v5; // r14
  __int64 v6; // r12
  char v7; // al
  __int64 i; // rdi
  __m128i *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // r8

  v4 = a2;
  if ( *(_BYTE *)(a2 + 140) != 12 )
    goto LABEL_5;
  do
    v4 = *(_QWORD *)(v4 + 160);
  while ( *(_BYTE *)(v4 + 140) == 12 );
  while ( *(_BYTE *)(a1 + 140) == 12 )
  {
    a1 = *(_QWORD *)(a1 + 160);
LABEL_5:
    ;
  }
  v5 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 40LL);
  if ( v5 )
  {
    v6 = v5[1];
    if ( v5 != a3 )
    {
      while ( v6 )
      {
        v7 = *(_BYTE *)(v6 + 80);
        if ( v7 == 16 )
        {
          v6 = **(_QWORD **)(v6 + 88);
          v7 = *(_BYTE *)(v6 + 80);
        }
        if ( v7 == 24 )
          v6 = *(_QWORD *)(v6 + 88);
        for ( i = *(_QWORD *)(*(_QWORD *)(v6 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v9 = sub_73D790(i);
        v11 = sub_6EEB30((__int64)v9, 0);
        if ( v4 == v11 || (unsigned int)sub_8D97D0(v4, v11, 0, v10, v12) )
          return v6;
        v5 = (_QWORD *)*v5;
        if ( v5 )
        {
          v6 = v5[1];
          if ( a3 != v5 )
            continue;
        }
        return 0;
      }
    }
  }
  return 0;
}
