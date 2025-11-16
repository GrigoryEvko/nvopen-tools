// Function: sub_2A9D270
// Address: 0x2a9d270
//
__int64 __fastcall sub_2A9D270(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // rsi
  _QWORD *v9; // rax
  _QWORD *v10; // rdx

  if ( a2 == a1 )
    return 0;
  v3 = a1;
  v6 = 0;
  v7 = a3 + 56;
  do
  {
    v8 = *(_QWORD *)(*(_QWORD *)(v3 + 24) + 40LL);
    if ( *(_BYTE *)(a3 + 84) )
    {
      v9 = *(_QWORD **)(a3 + 64);
      v10 = &v9[*(unsigned int *)(a3 + 76)];
      if ( v9 != v10 )
      {
        while ( v8 != *v9 )
        {
          if ( v10 == ++v9 )
            goto LABEL_9;
        }
        ++v6;
      }
    }
    else if ( sub_C8CA60(v7, v8) )
    {
      ++v6;
    }
    do
LABEL_9:
      v3 = *(_QWORD *)(v3 + 8);
    while ( v3 && (unsigned __int8)(**(_BYTE **)(v3 + 24) - 30) > 0xAu );
  }
  while ( a2 != v3 );
  return v6;
}
