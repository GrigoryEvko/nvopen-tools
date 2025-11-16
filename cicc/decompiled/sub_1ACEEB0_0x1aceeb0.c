// Function: sub_1ACEEB0
// Address: 0x1aceeb0
//
char __fastcall sub_1ACEEB0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // rsi
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 v8; // rbx
  __int64 i; // r13
  __int64 v10; // rsi

  v1 = *(_QWORD *)a1;
  v2 = *(_QWORD *)(*(_QWORD *)a1 + 16LL);
  v3 = *(_QWORD *)a1 + 8LL;
  if ( v2 != v3 )
  {
    do
    {
      v4 = v2 - 56;
      if ( !v2 )
        v4 = 0;
      sub_1ACEBA0(a1, v4);
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v3 != v2 );
    v1 = *(_QWORD *)a1;
  }
  v5 = *(_QWORD *)(v1 + 32);
  v6 = v1 + 24;
  if ( v1 + 24 != v5 )
  {
    do
    {
      v7 = v5 - 56;
      if ( !v5 )
        v7 = 0;
      sub_1ACEBA0(a1, v7);
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v6 != v5 );
    v1 = *(_QWORD *)a1;
  }
  v8 = *(_QWORD *)(v1 + 48);
  for ( i = v1 + 40; i != v8; v8 = *(_QWORD *)(v8 + 8) )
  {
    v10 = v8 - 48;
    if ( !v8 )
      v10 = 0;
    LOBYTE(v1) = sub_1ACEBA0(a1, v10);
  }
  return v1;
}
